from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, uuid, wandb
from accelerate import Accelerator

accelerator = Accelerator()

set_seed(42)

modelpath = "../models/llama3-8b"

# 3 reserved tokens -> im_start, im_end, pad
tokenizerpath = "../models/llama3-8b-chatML-tokenizer"

run_id = f"llama3-8b-slimhermes-{str(uuid.uuid4())}"

model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_quant_type = "nf4",
    ),
    attn_implementation = "flash_attention_2",  
    use_cache = False,
)
tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)

# model, tokenizer = setup_chat_format(model, tokenizer)
# if tokenizer.pad_token in [None, tokenizer.eos_token]: 
#     tokenizer.pad_token = tokenizer.unk_token

model.generation_config.bos_token_id = 128002
model.generation_config.eos_token_id = 128003
model.generation_config.pad_token_id = 128004
model.config.bos_token_id = 128002
model.config.eos_token_id = 128003
model.config.pad_token_id = 128004
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

dataset = load_dataset("g-ronimo/SlimHermes")

bs=1
ga=8
steps_per_epoch=len(dataset["train"])//(accelerator.state.num_processes*bs*ga)

training_arguments = TrainingArguments(
    output_dir = f"out_{run_id}",
    evaluation_strategy = "steps",
    label_names = ["labels"],
    per_device_train_batch_size = bs,
    gradient_accumulation_steps = ga,
    save_steps = steps_per_epoch,
    eval_steps = steps_per_epoch,
    logging_steps = 1, 
    learning_rate = 1e-5,
    num_train_epochs = 4,
    lr_scheduler_type = "constant",
    optim = 'paged_adamw_8bit',
    bf16 = True,
    gradient_checkpointing = True,
    group_by_length = True,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset['test'],
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template = "<|im_start|>user", 
        response_template = "<|im_start|>assistant", 
        tokenizer = tokenizer, 
        mlm = False),
    max_seq_length = 1_500,
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules = "all-linear", 
        modules_to_save = ["lm_head", "embed_tokens"]
    ),
    dataset_kwargs = dict(add_special_tokens = False),
    args = training_arguments,
    dataset_num_proc = 8,
)

for strs in [
    "<|im_start|>",
    "<|im_end|>",
    "<pad>",
]:
    print(strs, tokenizer(strs))

if accelerator.is_main_process:
    wandb.init(
        project = "slim-hermes-llama2-8b", 
        name = run_id,
    ).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()