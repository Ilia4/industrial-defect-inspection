import torch
import torch.nn as nn
import torch.optim as optim
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import DefectCNN

EPOCHS = 30

os.makedirs("./models", exist_ok=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

full_train_dataset = datasets.ImageFolder(
    "./HEAD/Train",
    transform=train_transform
)

full_val_dataset = datasets.ImageFolder(
    "./HEAD/Train",
    transform=val_transform
)

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(full_train_dataset),
    generator=generator
).tolist()

split = int(len(indices) * 0.8)

train_indices = indices[:split]
val_indices = indices[split:]

train_dataset = Subset(
    full_train_dataset,
    train_indices
)

val_dataset = Subset(
    full_val_dataset,
    val_indices
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

model = DefectCNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

best_val_accuracy = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device=device)
        labels = labels.to(device=device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total * 100

    print(
        f"Эпоха {epoch + 1} | "
        f"Loss: {average_loss} | "
        f"Validation: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "./models/defect_cnn.pth"
        )

        print("Сохранена лучшая модель")
