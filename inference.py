from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch, gc
import torch.nn as nn

class ModelPredictionGenerator:
    # llama-precise as default, from https://github.com/oobabooga/text-generation-webui/blob/main/presets/LLaMA-Precise.yaml
    DEFAULT_GEN_CONFIG={
            "temperature": 0.7,
            "top_p": 0.1,
            "repetition_penalty": 1.18,
            "top_k": 40,
            "do_sample": True,
            "max_new_tokens": 50,
        }

    def Hermes_to_HF(input):
        roles_map= {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
        }
        messages=[dict(role=roles_map[msg["from"]], content=msg["value"]) for msg in input["conversations"]]
        return dict(messages=messages)

    def Orca_to_HF(input):
        output = [
            dict(role="user", content=input["question"]),
            dict(role="assistant", content=input["response"]),
        ]
        if len(input["system_prompt"])>0:
            output.insert(0, dict(role="system", content=input["system_prompt"]))
        return dict(messages=output)

    def Ultrachat_to_HF(input):
        return dict(messages=input["messages"])

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
            dataset has to have column "messages" = 
            [ 
                {"role": "user", "content": "Hello, how are you?"}, 
                {"role": "assistant", "content": "I am ..?"}, 
                ..
            ]  
            the method iterates over messages and whenever a message by user is encountered, it uses the tokenizer to format all messages up to and including the current one and adds it to the returned prompts
        """
        prompts = []
        for conversation in ds["messages"]:
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
        with torch.cuda.amp.autocast(), torch.no_grad():
            outputs_tok = self.model.generate(
                input_ids = batch_tok["input_ids"],
                attention_mask = batch_tok["attention_mask"],
                **generation_config
            ).to("cpu")
        timediff=time.time() - start_time

        # cut prompt from output
        outputs=[ 
            self.tokenizer.decode(tok_out[len(tok_in):], 
                spaces_between_special_tokens = False, 
                skip_special_tokens=False).strip() 
            for tok_in, tok_out in zip(batch_tok["input_ids"], outputs_tok) ] 
        outputs_tokencount = torch.sum(outputs_tok!=self.tokenizer.pad_token_id).item()

        return outputs, outputs_tokencount//timediff

    def run(self, dataset, generation_config=None, batch_size=8):
        """  extracts prompts from dataset, generates answers using batched inference  """
        generation_config = ModelPredictionGenerator.DEFAULT_GEN_CONFIG if generation_config is None else generation_config
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        self.model.eval()
        self.clear_cache()

        prompts = self.messages_to_prompts( dataset )
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        for batch in tqdm(batches):
            batch_tok = self.tokenize_batch( batch )
            answers, tok_per_second = self.generate_batch( batch_tok, generation_config )       
            for prompt, answer in zip(batch, answers): 
                prompt["answer_pred"] = answer
                prompt["toks/s"] = tok_per_second

        return prompts

class ModelPredictionGeneratorDistributed(ModelPredictionGenerator):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.accelerator = Accelerator()

    def messages_to_prompts(self, ds):
        """ distributed prompts to all GPUs, each then works on its own local subset of prompts """
        prompts = super().messages_to_prompts(ds)
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

