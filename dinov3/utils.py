import torch
from datasets import load_dataset
from functools import partial

def collate_fn(rows, processor):
    labels = [r["label"] for r in rows]
    images = [r["image"] for r in rows]
    tensors_list = [processor(images=img, return_tensors="pt")["pixel_values"] for img in images]
    tensor = torch.cat(tensors_list)
    
    return tensor, labels

def load_fooddataset(processor, batch_size_train, batch_size_test=None):
    ds = load_dataset("johnowhitaker/imagenette2-320")
    ds = ds["train"].shuffle().train_test_split(test_size=0.1)
    if batch_size_test is None:
        batch_size_test = batch_size_train // 2
    
    dataloader_train = torch.utils.data.DataLoader(
        ds["train"], 
        batch_size = batch_size_train,
        shuffle = True,
        collate_fn = collate_fn,
        prefetch_factor = 2,
        num_workers = 4,
    )

    dataloader_test = torch.utils.data.DataLoader(
        ds["test"], 
        batch_size = batch_size_test,
        # shuffle = True,
        collate_fn = collate_fn,
        prefetch_factor = 2,
        num_workers = 4,
    )

    return dataloader_train, dataloader_test

from datasets import concatenate_datasets

def load_nsfwdataset(processor, batch_size_train, batch_size_test=None, seed=42):
    if batch_size_test is None:
        batch_size_test = batch_size_train // 2

    # Load NOT SFW
    ds_nsfw = load_dataset("yesidobyte/nsfw1024")["train"]
    # Load SFW = Imagenet
    ds_sfw = load_dataset("johnowhitaker/imagenette2-320")["train"]

    # Mix them equally: `mixed_dataset`
    min_size = min(len(ds_nsfw), len(ds_sfw))
    ds_nsfw = ds_nsfw.shuffle(seed=seed).select(range(min_size)).map(lambda x: {"label": 1})
    
    # Resize nsfw
    ds_nsfw = ds_nsfw.map(
        lambda x: {"image": x["image"].resize((320, 320))},
        batched = False,
        num_proc = 8,
        desc = "Resizing NSFW images"
    )
    ds_sfw = ds_sfw.remove_columns("label").shuffle(seed=seed).select(range(min_size)).map(lambda x: {"label": 0})
    mixed_dataset = concatenate_datasets([ds_nsfw, ds_sfw])

    # Shuffle and split into train and test: `ds`
    ds = mixed_dataset.train_test_split(test_size=0.1, seed=seed)

    # Prepare dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        ds["train"], 
        batch_size = batch_size_train,
        shuffle = True,
        collate_fn = partial(collate_fn, processor=processor),
        prefetch_factor = 2,
        num_workers = 4,
    )

    dataloader_test = torch.utils.data.DataLoader(
        ds["test"], 
        batch_size = batch_size_test,
        # shuffle = True,
        collate_fn = partial(collate_fn, processor=processor),
        prefetch_factor = 2,
        num_workers = 4,
    )

    return dataloader_train, dataloader_test


from tqdm import tqdm

def test_accuracy(dataset, model):
    num_correct, num_total = 0, 0
    
    for images, labels in tqdm(dataset, "Test accuray"):
        with torch.no_grad():
            logits = model(images.to(model.device))
        predictions = torch.argmax(logits, dim=1).cpu()
        
        num_correct += (torch.Tensor(labels) == predictions.cpu()).sum().item()
        num_total += len(labels)

    return num_correct / num_total * 100