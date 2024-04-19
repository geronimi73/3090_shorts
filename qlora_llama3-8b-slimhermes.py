from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, set_seed, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, uuid, wandb
from accelerate import Accelerator

set_seed(42)
accelerator = Accelerator()
run_id = f"llama3-8b-slimhermes-{str(uuid.uuid4())}"
modelpath = "../models/llama3-8b"
tokenizerpath = "../models/llama3-8b-chatML-tokenizer" # slightly modified tokenizer: 3 reserved tokens -> im_start, im_end, pad
bs, ga = (1, 8)  # batch size, grad. acc. steps
epochs = 4
lr = 1e-5
max_seq_length = 1800

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

tokenizer = AutoTokenizer.from_pretrained(tokenizerpath, use_fast=False)

## old
# template = llama3.bos + ChatML + llama3.eos; if no bos/eos present -> grad=nan 
# tokenizer.chat_template = "{{ '<|begin_of_text|>' }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{{ '<|end_of_text|>' }}"

# ChatML template
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

# set embeddings of <|im_start|>=128002 and <|im_end|>=128003 to avg. of all other tokens= id<128000 (from artidoro/qlora: smart embedding resize)
tokens_ids = [128_002, 128_003]
input_embeddings_data = model.get_input_embeddings().weight.data
output_embeddings_data = model.get_output_embeddings().weight.data

input_embeddings_avg = input_embeddings_data[0:128_000].mean(dim=0, keepdim=True)
output_embeddings_avg = output_embeddings_data[0:128_000].mean(dim=0, keepdim=True)

for t in tokens_ids:
    input_embeddings_data[t] = input_embeddings_avg
    output_embeddings_data[t] = output_embeddings_avg

dataset = load_dataset("g-ronimo/SlimHermes")

steps_per_epoch=len(dataset["train"])//(accelerator.state.num_processes*bs*ga)

training_arguments = TrainingArguments(
    output_dir = f"out_{run_id}",
    evaluation_strategy = "steps",
    label_names = ["labels"],
    per_device_train_batch_size = bs,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = ga,
    save_steps = steps_per_epoch,
    eval_steps = steps_per_epoch,
    logging_steps = 1, 
    learning_rate = lr,
    num_train_epochs = epochs,
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
    max_seq_length = max_seq_length,
    peft_config = LoraConfig(
        r=32, 
        lora_alpha=32, 
        target_modules = "all-linear", 
        modules_to_save = ["lm_head", "embed_tokens"]
    ),
    dataset_kwargs = dict(add_special_tokens = False),
    args = training_arguments,
    dataset_num_proc = 8,
)

## Eval after 1st step to check if eval possible w/o OOM
# class EvaluateFirstStepCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         if state.global_step == 1:
#             control.should_evaluate = True
# trainer.add_callback(EvaluateFirstStepCallback())

## Check correct tokenization and labeling of samples
check_samples = 1
for i in range(check_samples):
    print(f"SAMPLE #{i}")
    input_ids, attention_mask, labels = trainer.data_collator([trainer.train_dataset[0]]).values()
    print(tokenizer.decode(input_ids[0]))  
    for token, label in zip(input_ids[0], labels[0]):
        print(f"{token.item()}, '{tokenizer.decode(token)}', {label.item()}")  
    print()

if accelerator.is_main_process:
    wandb.init(
        project = "slim-hermes-llama2-8b", 
        name = run_id,
    ).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()