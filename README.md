# nb_QLoRA
QLoRA finetune example for llama2-7b and Open Assistant 2 dataset 

# inference.py
## ModelPredictionGenerator
```python
model, tokenizer = ..
dataset = load_dataset("g-ronimo/oasst2_top4k_en")["test"]

generator = ModelPredictionGeneratorDistributed(
    model = model,
    tokenizer = tokenizer,
)
results = generator.run(
    dataset = eval_ds,
    batch_size = 2,
)
```

## ModelPredictionGeneratorDistributed
same as ModelPredictionGenerator for multi GPU inference with accelerate 
