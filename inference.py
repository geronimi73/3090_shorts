from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch, gc
import torch.nn as nn

class ModelPredictionGenerator:
    def __init__(self, model, tokenizer, eval_dataset, use_accelerate=False, bs=8, generation_config=None):
        self.model=model
        self.tokenizer=tokenizer
        self.bs=bs
        self.eval_prompts=self.messages_to_prompts( eval_dataset )
        self.use_accelerate=use_accelerate
        self.accelerator = Accelerator()

        assert tokenizer.eos_token_id is not None
        assert tokenizer.chat_template is not None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # llama-precise
        if generation_config is None:            
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "do_sample": True,
                "max_new_tokens": 100,
                "pad_token_id": tokenizer.pad_token_id
            }
        else:
            self.generation_config = generation_config

    def clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def messages_to_prompts(self, ds):
        prompts=[]
        for conversation in ds["messages"]:
            for i,msg in enumerate(conversation):
                if msg["role"]=="user":
                    prompts.append(
                        dict (
                            # prompt: format current messages up to the current user message and add a generation prompt
                            prompt=self.tokenizer.apply_chat_template(conversation[:i+1], add_generation_prompt=True, tokenize=False),
                            answer_ref=conversation[i+1]["content"]
                        )
                    )
        return prompts

    def get_batches(self, dataset, batch_size):
        return [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]  

    def tokenize_batch(self, batch):
        pad_side=self.tokenizer.padding_side
        self.tokenizer.padding_side="left"     # left pad for inference
        
        prompts=[ item["prompt"] for item in batch ]   
        prompts_tok=self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_length=True,
            pad_to_multiple_of=8,
            add_special_tokens=False
        ).to(self.model.device)
        self.tokenizer.padding_side=pad_side   # restore orig. padding side
        
        return prompts_tok

    def generate_batch(self, batch_tok):
        with torch.no_grad():
            outputs_tok=self.model.generate(
                input_ids=batch_tok["input_ids"],
                attention_mask=batch_tok["attention_mask"],
                **self.generation_config
            ).to("cpu")
        outputs=[
            # cut prompt from output
            self.tokenizer.decode(
                outputs_tok[i][outputs_tok[i] != self.tokenizer.pad_token_id][batch_tok["length"][i]:], 
                spaces_between_special_tokens=False,
                skip_special_tokens=True
                ).strip()
            for i,t in enumerate(outputs_tok) ]

        return outputs

    def run(self):
        self.model.eval()
        self.clear_cache()
    
        if self.use_accelerate:
            with self.accelerator.split_between_processes(list(range(len(self.eval_prompts)))) as eval_prompts_local_idcs:
                eval_prompts_local = [self.eval_prompts[i] for i in eval_prompts_local_idcs]
        else:
            eval_prompts_local = self.eval_prompts

        for batch in tqdm( self.get_batches(eval_prompts_local, self.bs) ):
            batch_tok = self.tokenize_batch( batch )
            answers = self.generate_batch( batch_tok )   
    
            for i in range(len(batch)):
                batch[i]["answer_pred"]=answers[i]
                batch[i]["GPU"]=self.accelerator.process_index
            
        if self.use_accelerate:
            return gather_object(eval_prompts_local)
        else:
            return eval_prompts_local