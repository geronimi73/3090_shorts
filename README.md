# 3090 shorts
A bunch of minimal notebooks and scripts I use for training and inference, designed to run on NVIDIA GPUs with 24 GB VRAM like a RTX 3090 or 4090.

## Tools in `utils.py`
### ModelPredictionGenerator
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
Same as `ModelPredictionGenerator` but for multi-GPU inference with HF accelerate.

### EmbeddingModelWrapper
Calculate embedding vectors and cosine similarities of a list of strings; default embedding model is `sentence-transformers/all-mpnet-base-v2`.

```python
from utils import EmbeddingModelWrapper
em = EmbeddingModelWrapper()

words = ["lemon", "orange", "car", "money"]
embds = em.get_embeddings(words)

similarities = em.get_similarities(embds)
``` 
### SingleChoiceEval
Calculate accuracy of a given model on a single-choice dataset.

**MMLU**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

modelpath = "models/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    torch_dtype = torch.bfloat16,
    device_map = "auto",
    attn_implementation = "flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False) 
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

dataset = load_dataset("cais/mmlu", "all")

from utils import SingleChoiceEval

sce = SingleChoiceEval(dataset["dev"])
total, correct, acc = sce.calc_accuracy(
	model, 
	tokenizer, 
	batch_size = 16
)
```
Output TinyLlama
```
(285, 66, 23.157894736842106)
```

**PIQA 5-shot**
```python
# load model and tokenizer just like before
...

from utils import SingleChoiceEval
from datasets import load_dataset

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

dataset = load_dataset("piqa")

sce = SingleChoiceEval(
    dataset["validation"], 
    key_choices = ['sol1', 'sol2'],
    key_question = "goal",
    key_answer = "label"
)
total, correct, acc = sce.calc_accuracy(
    model, 
    tokenizer, 
    few_shots = dataset["train"].select(range(5)),
    batch_size = 16,
)
```
Output Mistral-0.2 (base):
```
(1838, 1474, 80.19586507072906)
```

**Kaggle's LLM Science Exam**
```python
# load model and tokenizer just like before
...

# this part is new
from datasets import load_dataset
from utils import SingleChoiceEval

dataset = load_dataset("g-ronimo/kaggle_llm_science_exam")

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

sce = SingleChoiceEval(
    dataset["test"], 
    key_choices = ['A', 'B', 'C', 'D', 'E'],
    key_question = "prompt"
)
total, correct, acc = sce.calc_accuracy(
    model, 
    tokenizer, 
    batch_size = 16
)
```
Output TinyLlama
```
(600, 135, 22.5)
```
