# Notebooks for training/inference `nb_*`

# Tools in `utils.py`
## ModelPredictionGenerator
Little helper for batch inference, see `nb_batch-inference.ipynb` for usage or this:

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

### ModelPredictionGeneratorDistributed
same as ModelPredictionGenerator for multi GPU inference with accelerate 

## EmbeddingModelWrapper
calculate embedding vectors and cosine similarities of a list of strings; default embedding model is `sentence-transformers/all-mpnet-base-v2`

```python
from utils import EmbeddingModelWrapper
em = EmbeddingModelWrapper()

words = ["lemon", "orange", "car", "money"]
embds = em.get_embeddings(words)

similarities = em.get_similarities(embds)
``` 
## SingleChoiceEval
Calculate accuracy of a given model on a dataset
### MMLU
 Code
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

modelpath="models/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False) 

dataset = load_dataset("cais/mmlu", "all")

from utils import SingleChoiceEval
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

sce = SingleChoiceEval(dataset["dev"])
total, correct, acc = sce.calc_accuracy(
	model, 
	tokenizer, %%  %%
	batch_size=16
)
```
Output
```
(285, 66, 23.157894736842106)
```