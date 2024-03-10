# Notebooks

## nb_QLoRA_*
QLoRA finetune minimal scripts

## nb_finetune-full_*
full finetune minimal scripts

# utils.py
## ModelPredictionGenerator
little hepler for batch inference, see `nb_batch-inference.ipynb` for usage or this:

```python
model, tokenizer = ..
dataset = load_dataset("g-ronimo/oasst2_top4k_en")["test"]

generator = ModelPredictionGeneratorDistributed(
    model = model,
    tokenizer = tokenizer,
)
results = generator.run(
    input_data = eval_ds,
    batch_size = 2,
)
```

## ModelPredictionGeneratorDistributed
same as ModelPredictionGenerator for multi GPU inference with accelerate 

## EmbeddingModelWrapper
calculate embedding vectors and cosine similarities of a list of strings; default embeddding model is `sentence-transformers/all-mpnet-base-v2`

```python
from utils import EmbeddingModelWrapper
em = EmbeddingModelWrapper()

words = ["lemon", "orange", "car", "money"]
embds = em.get_embeddings(words)

similarities = em.get_similarities(embds)
``` 
