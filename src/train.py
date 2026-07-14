import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloaders
from model import build_model, unfreeze_last_block

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def main():
    train_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    print("\n--- Stage 1: training head ---")
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    model = unfreeze_last_block(model)
    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-5},
    ])
    print("\n--- Stage 2: fine-tuning last block ---")
    for epoch in range(5):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), "../models/emotion_model.pth")
    print("Model saved to models/emotion_model.pth")

if __name__ == "__main__":
    main()