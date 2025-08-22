import json
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_loaders(data_dir="data", img_size=224, batch_size=32):
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(Path(data_dir)/"train", transform=train_tfm)
    val_ds   = datasets.ImageFolder(Path(data_dir)/"val",   transform=val_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.class_to_idx

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item()*y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum/total, correct/total

def train(data_dir="data", epochs=5, lr=1e-3, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, class_to_idx = get_loaders(data_dir, batch_size=batch_size)

    num_classes = len(class_to_idx)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False  
    model.fc = nn.Linear(model.fc.in_features, num_classes)  
    model = model.to(device)

    optimizer = optim.AdamW(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()*y.size(0)
            total += y.size(0)

        train_loss = running/total
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model.pth")
            print("Saved best model to model.pth")

    
    with open("class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print("Saved class mapping to class_to_idx.json")

if __name__ == "__main__":
    train(data_dir="data", epochs=8, lr=1e-3, batch_size=32)
