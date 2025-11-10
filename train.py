import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2
from transformer import ViT

device = torch.device("mps")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@torch.no_grad()
def score(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device) -> dict:
    
    model.eval()
    val_loss = 0.0
    correct_pred = 0
    n_datapoints = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        with autocast(device.type):
            logits = model(imgs)
            loss = criterion(logits, targets)
        val_loss += loss
        targets = targets.view(-1, 1)
        pred = logits.topk(1, dim=1, largest=True)[1]
        correct_pred += pred.eq(targets).sum().item()
        n_datapoints += imgs.size(0)
        if n_datapoints > 500:
            break


    acc = correct_pred / n_datapoints

    return {"acc": acc, "loss": val_loss}

if __name__ == '__main__':
    
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomPhotometricDistort(),
        v2.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform = transformer)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform = transformer)


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
        embed_dim = 150,
        depth = 4,
        nheads = 6,
        mlp_ratio = 3,
        dropout = 0.1,
        num_classes = len(train.classes)
    ).to(device)

    print(model)
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,}")

    compiled_model = torch.compile(model, backend="aot_eager")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler('mps')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,               # peak LR
        total_steps=len(train_loader) * 100,
        pct_start=0.1,             # 10% of steps = warm-up
        anneal_strategy='cos',
        div_factor=25.0,           # initial_lr = max_lr / 25
        final_div_factor=1e4       # final_lr = initial_lr / 1e4
    )   

    for epoch in range(100):
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
            scheduler.step()
            scaler.update()
            optimizer.zero_grad()
            step += 1
            print(f"step[{step}] | batch loss[{loss.item():.2f}] | lr[{optimizer.param_groups[0]['lr']}]")
            total_loss += loss.item()

        print(f"Epoch[{epoch}]: {total_loss}")
        val_results = score(model, criterion, test_loader, device)
        train_results = score(model, criterion, train_loader, device) 
        print(f"Val acc: {val_results['acc']:.4f}, Val loss: {val_results['loss']:.4f}")
        print(f"Train acc: {train_results['acc']:.4f}, Train loss: {train_results['loss']:.4f}")
