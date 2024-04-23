from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from tqdm import tqdm
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
            y = x
        if x.dim()==y.dim()==1:
            return self.cos(x[None, :],y[None, :]).cpu().tolist()        
    
        x_num, y_num = x.shape[0], y.shape[0]
        similarities = torch.zeros(x_num, y_num)
        for row in tqdm(range(x_num)):
            similarities[row, :]=self.cos(x[row].repeat(y_num,1), y)
    
        return similarities

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
            prompts = self.messages_to_prompts(input_data)
        else:
            # list of strings hopefully
            prompts = self.questions_to_prompts(input_data)
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

class SingleChoiceEval:
    DEFAULT_TEMPLATE = "{question}\n{choices}\nAnswer:"
    LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    def __init__(self, dataset, template=DEFAULT_TEMPLATE, key_question="question", key_choices="choices", key_answer="answer"):
        self.dataset = dataset
        self.template = DEFAULT_TEMPLATE if template is None else template
        self.key_question = key_question
        self.key_choices = key_choices
        self.key_answer = key_answer
        self.answer_is_int = type(dataset[0][key_answer]) == int
        if type(self.key_choices) == list:
            self.num_choices = len(self.key_choices)
        else:
            self.num_choices = len(dataset[0][self.key_choices])

    def get_choices(self, entry):
        if type(self.key_choices) == list:
            return [entry[k] for k in self.key_choices]
        else:
            return entry[self.key_choices]    

    def get_answer(self, entry):
        if self.answer_is_int:
            # answer is int
            return entry[self.key_answer]
        elif not self.answer_is_int and entry[self.key_answer] in self.LETTERS:
            # answer is a string A-Z 
            return self.LETTERS.index(entry[self.key_answer])
        else:
            return None

    def format_entry(self, entry, include_answer = True):
        template = self.template

        choices = [ f"{self.LETTERS[i]}. {choice}" for i, choice in enumerate(self.get_choices(entry)) ]
        choices =  "\n".join(choices)
        text = template.format(choices = choices, question = entry[self.key_question])
        
        if include_answer:
            text += f" {self.LETTERS[self.get_answer(entry)]}"
        return text

    def calc_accuracy(self, model, tokenizer, batch_size = 8, few_shots = None):
        choices_tok = [ 
            tokenizer(self.LETTERS[i], add_special_tokens = False)["input_ids"][-1] 
            for i in range(self.num_choices) 
        ]
        
        if few_shots is not None:
            few_shot_prompt = []
            for entry in few_shots:
                few_shot_prompt.append(self.format_entry(entry, include_answer = True))
            few_shot_prompt = "\n\n".join(few_shot_prompt) + "\n\n"
        else:
            few_shot_prompt = ""
        
        questions = [ 
            few_shot_prompt + self.format_entry(entry, include_answer = False) 
            for entry in self.dataset
        ]

        # debug
        for i in range(3):
            print(f"Question #{i}")
            print(questions[i], "\n")
            
        batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]  
        
        total, correct = 0, 0    
        with tqdm(total = len(batches)) as pbar:
            for batch_no, batch in enumerate(batches):
                pbar.update()
                batch_tok = tokenizer(batch, return_tensors = "pt", padding = True).to("cuda")
    
                with torch.no_grad():
                    batch_logits = model(**batch_tok).logits
                    # batch_logits.to("cpu")
                    
                for i, logits in enumerate(batch_logits):
                    model_choice = torch.argmax(logits[-1][choices_tok]).item()  # -1 is last logit, choices_tok = logits for A, B, C, D
                    correct += 1 if model_choice == self.get_answer(self.dataset[total]) else 0
                    total += 1
                pbar.set_postfix_str(f"acc={round(correct/total*100,2)}")
        return total, correct, correct / total * 100
