from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
import torch, torch.nn as nn, gc
import time

class EmbeddingModelWrapper():
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_path=DEFAULT_MODEL, bs=8):
        self.model, self.tokenizer = self.load_model(model_path)
        self.bs = bs
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_model(self, model_path):
        model = AutoModel.from_pretrained(model_path).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model.eval(), tokenizer

    def emb_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        assert isinstance(sentences, list), f"sentences has to be a list but is {type(sentences)}"

        embeddings=torch.tensor([], device = "cuda")
        batches = [sentences[i:i + self.bs] for i in range(0, len(sentences), self.bs)] if self.bs else [sentences]
        for sentences in tqdm(batches):
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to("cuda")
            with torch.no_grad():
                model_output = self.model(**encoded_input)        
            batch_embeddings = self.emb_mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings=torch.cat( (embeddings, batch_embeddings), dim=0 )
        return embeddings

    def get_similarities(self, x, y=None):
        assert isinstance(x, torch.Tensor), f"x has to be a Tensor but is {type(x)}"

        if y is None:
            num_samples=x.shape[0]
            similarities = torch.zeros(num_samples, num_samples)
            for row in tqdm(range(num_samples)):
                similarities[row, 0:row+1]=self.cos(x[row].repeat(row+1,1), x[0:row+1])#.tolist()

            similarities = (similarities + similarities.T)
            similarities.diagonal()[:] = 1

            return similarities
        else:            
            assert isinstance(y, torch.Tensor), f"y has to be a Tensor but is {type(y)}"
            return self.cos(x,y).cpu().tolist()

class ModelPredictionGenerator:
    # llama-precise as default, from https://github.com/oobabooga/text-generation-webui/blob/main/presets/LLaMA-Precise.yaml
    DEFAULT_GEN_CONFIG={
            "temperature": 0.7,
            "top_p": 0.1,
            "repetition_penalty": 1.18,
            "top_k": 40,
            "do_sample": True,
            # "max_new_tokens": 500,
        }

    def __init__(self, model, tokenizer):
        assert tokenizer.eos_token_id is not None
        assert tokenizer.chat_template is not None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.model = model
        self.tokenizer = tokenizer

    def clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def messages_to_prompts(self, ds):
        """  extract formatted prompts from dataset 
            dataset has to have column a "messages" or "conversation" = 
            [ 
                {"role": "user", "content": "Hello, how are you?"}, 
                {"role": "assistant", "content": "I am ..?"}, 
                ..
            ]  
            the method iterates over messages and whenever a message by user is encountered, it uses the tokenizer to format all messages up to and including the current one and adds it to the returned prompts
        """
        conversations = ds["messages"] if "messages" in ds.features else ds["conversation"]
        prompts = []
        for conversation in conversations:
            for i, msg in enumerate(conversation):
                if msg["role"] == "user":
                    prompts.append(
                        dict (
                            # prompt: format current messages up to the current user message and add a generation prompt
                            prompt = self.tokenizer.apply_chat_template(
                                conversation[:i+1], 
                                add_generation_prompt = True, 
                                tokenize = False
                                ),
                            answer_ref = conversation[i+1]["content"]
                        )
                    )
        return prompts

    def questions_to_prompts(self, questions):
        prompts = []
        for q in questions:
            messages = [dict(role="user", content=q)]
            prompts.append(
                dict (
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt = True, 
                        tokenize = False
                    ),
                )
            )
        return prompts

    def tokenize_batch(self, batch):
        """ tokenizes a list of prompts, returns a padded tensor """
        pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"     # left pad for inference
        
        prompts = [ item["prompt"] for item in batch ]   
        prompts_tok = self.tokenizer(
            prompts, 
            return_tensors = "pt", 
            padding = 'longest', 
            truncation = True,
            max_length = min(self.tokenizer.model_max_length, 1024),
            return_length = True,
            pad_to_multiple_of = 8,
            add_special_tokens = False
        ).to(self.model.device)
        self.tokenizer.padding_side = pad_side   # restore orig. padding side
        
        return prompts_tok

    def generate_batch(self, batch_tok, generation_config):
        """ generate prediction with batches of tokenized prompts, returns newly generated output only, without prompt """
        start_time = time.time()
        # Mistral trouble https://github.com/huggingface/peft/issues/1515
        # with torch.cuda.amp.autocast(), torch.no_grad():
        with torch.no_grad():
            outputs_tok = self.model.generate(
                input_ids = batch_tok["input_ids"],
                attention_mask = batch_tok["attention_mask"],
                **generation_config
            ).to("cpu")
        timediff=time.time() - start_time

        # cut prompt from output
        outputs_tok=[ tok_out[len(tok_in):] for tok_in, tok_out in zip(batch_tok["input_ids"], outputs_tok) ] 
        outputs=[ self.tokenizer.decode(tok, 
                spaces_between_special_tokens = False, 
                skip_special_tokens=True
                ).strip() 
            for tok in outputs_tok ] 
        outputs_tokencount = sum([len(o) for o in outputs_tok])

        return outputs, outputs_tokencount // timediff

    def input_to_prompts(self, input_data):
        assert isinstance(input_data, Dataset) or isinstance(input_data, list)

        if isinstance(input_data, Dataset):
            prompts = self.messages_to_prompts( input_data )
        else:
            prompts = self.questions_to_prompts( input_data )
        return prompts

    def run(self, input_data, generation_config=None, batch_size=64, max_new_tokens=500):
        """  generates prompts from datasets.Dataset or list of strings, generates answers using batched inference  """

        generation_config = ModelPredictionGenerator.DEFAULT_GEN_CONFIG if generation_config is None else generation_config
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        if max_new_tokens is not None:
            generation_config["max_new_tokens"] = max_new_tokens

        self.model.eval()

        while batch_size > 0:
            self.clear_cache()

            prompts = self.input_to_prompts( input_data )
            batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
            try:
                for batch in tqdm(batches):
                    batch_tok = self.tokenize_batch( batch )
                    answers, tok_per_second = self.generate_batch( batch_tok, generation_config )       
                    for prompt, answer in zip(batch, answers): 
                        prompt["answer_pred"] = answer
                        prompt["tok/s"] = tok_per_second
                return prompts
            except torch.cuda.OutOfMemoryError as e:
                batch_size //= 2
                print("OOM, retrying with batch size", batch_size)
        print("Failed due to OOM, not enough VRAM to generate even with a batch_size of 1")


class ModelPredictionGeneratorDistributed(ModelPredictionGenerator):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.accelerator = Accelerator()

    def input_to_prompts(self, input_data):
        """ distributed prompts to all GPUs, each then works on its own local subset of prompts """
        prompts = super().input_to_prompts(input_data)
        prompts_idcs = list(range(len(prompts)))
        with self.accelerator.split_between_processes(prompts_idcs) as prompts_idcs_local:
            prompts_local = [prompts[i] for i in prompts_idcs_local]
        return prompts_local

    def run(self, **kwargs):
        """ process local subset of prompts and gathers the results from all GPUs """
        results = super().run(**kwargs)
        for i in range(len(results)):
            results[i]["GPU"] = self.accelerator.process_index 
        return gather_object(results)
