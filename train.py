import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler

from transformer import ViT

device = torch.device("mps")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == '__main__':
    
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train = datasets.CIFAR100(root='./data', train=True, download=True, transform = transformer)
    test = datasets.CIFAR100(root='./data', train=False, download=True, transform = transformer)


    train_loader = DataLoader(
        train,
        batch_size = 128,
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
        drop_last = True,
        persistent_workers = True
    )

    test_loader = DataLoader(      
        test,
        batch_size = 128,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        drop_last = True,
        persistent_workers = True
    )

    model = ViT(
        img_size = 224,
        patch_size = 16,
        in_chans = 3,
        embed_dim = 300,
        depth = 3,
        nheads = 12,
        mlp_ratio = 1.0,
        dropout = 0.0,
        num_classes = len(train.classes)
    ).to(device)

    print(model)
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,}")

    compiled_model = torch.compile(model, backend="aot_eager")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler('mps')

    for epoch in range(10):
        model.train()
        step = 0
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to('mps', non_blocking=True)
            targets = targets.to('mps', non_blocking=True)
            with autocast('mps'):
                logits = compiled_model(imgs)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            print(f"step[{step}], batch loss[{loss.item()}]")
            total_loss += loss.item()

        print(f"Epoch[{epoch}]: {total_loss}")
