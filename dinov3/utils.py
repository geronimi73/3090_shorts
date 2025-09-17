import torch, torch.nn as nn
from datasets import load_dataset
from functools import partial

class DinoV3Linear(nn.Module):
    def __init__(self, backbone, hidden_size: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0]
        logits = self.head(cls)
        return logits

    def count_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    @property
    def device(self):
        return next(self.parameters()).device

def collate_fn(rows, processor, augment_fn=None):
    labels = [r["label"] for r in rows]
    images = [r["image"].convert('RGB') for r in rows]
    
    images = [augment_fn(img) for img in images] if augment_fn else images
    tensors_list = [processor(images=img, return_tensors="pt")["pixel_values"] for img in images]
    tensor = torch.cat(tensors_list)
    
    return tensor, labels, images

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
import torchvision.transforms as T
import os 

def augment(image, resizeTo=320):
    augmentation = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.CenterCrop((resizeTo, resizeTo)),
    ])
    
    image = image.resize((
        round(resizeTo * 1.2),
        round(resizeTo * 1.2),
    ))    
    image = augmentation(image)

    return image

def load_nsfwdataset(processor, batch_size_train, batch_size_test=None, seed=42, resizeTo=320):
    if batch_size_test is None:
        batch_size_test = batch_size_train // 2

    # Load NOT SFW
    ds_nsfw = load_dataset("yesidobyte/nsfw1024")["train"]

    # Resize and label nsfw
    ds_nsfw = ds_nsfw.map(
        lambda x: {
            "image": x["image"].resize((resizeTo, resizeTo)),
            "label": 1
        },
        batched = False,
        num_proc = os.cpu_count(),
        desc = "Resizing NSFW images"
    )

    # Load SFW = Imagenet + CelebA
    num_to_pick = len(ds_nsfw) 
    ds_sfw_premix = [
        load_dataset("johnowhitaker/imagenette2-320")["train"].remove_columns("label"),
        load_dataset("nielsr/CelebA-faces")["train"],
        load_dataset("yuvalkirstain/pexel_people")["train"].remove_columns(['text', 'generated_caption']),
    ]
    ds_sfw = concatenate_datasets([
        d.shuffle().select(range(min(num_to_pick // len(ds_sfw_premix)  + 10, len(d))))
        for d in ds_sfw_premix
    ]).shuffle()

    ds_sfw = ds_sfw.map(
        lambda x: {
            "image": x["image"].resize((resizeTo, resizeTo)),
            "label": 0
        },
        batched = False,
        num_proc = os.cpu_count(),
        desc = "Resizing SFW images"
    )

    mixed_dataset = concatenate_datasets([ds_nsfw, ds_sfw])
    print(f"SFW: {len(ds_sfw)} images, NSFW: {len(ds_nsfw)} images")

    # Shuffle and split into train and test: `ds`
    ds = mixed_dataset.train_test_split(test_size=0.1, seed=seed)

    # Prepare dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        ds["train"], 
        batch_size = batch_size_train,
        shuffle = True,
        collate_fn = partial(collate_fn, processor=processor, augment_fn=augment),
        prefetch_factor = 2,
        num_workers = 4,
    )

    dataloader_test = torch.utils.data.DataLoader(
        ds["test"], 
        batch_size = batch_size_test,
        # shuffle = True,
        collate_fn = partial(collate_fn, processor=processor, augment_fn=augment),
        prefetch_factor = 2,
        num_workers = 4,
    )

    return dataloader_train, dataloader_test


from tqdm import tqdm

def test_accuracy(dataset, model):
    num_correct, num_total = 0, 0
    test_loss = 0.0
    
    for images, labels, images_pil in tqdm(dataset, "Test accuray"):
        images, labels = images.to(model.device), torch.Tensor(labels).to(model.device).long()

        with torch.no_grad():
            logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        predictions = torch.argmax(logits, dim=1).cpu()
        
        test_loss += loss.item() * len(labels)
        num_correct += (labels.cpu() == predictions).sum().item()
        num_total += len(labels)

    return num_correct / num_total * 100, test_loss / num_total
