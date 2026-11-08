import torch
import torch.nn as nn
import torch.optim as optim
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DefectCNN

EPOCHS = 10

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

model = DefectCNN()

model.load_state_dict(
    torch.load(
        "./models/defect_cnn.pth",
        map_location=device
    )
)

model.linear = nn.Linear(in_features=28800, out_features=2)

for param in model.conv1.parameters():
    param.requires_grad = False

model = model.to(device=device)

train_dataset = datasets.ImageFolder(
    "./real_data",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "./validation",
    transform=val_transform
)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print(val_dataset.class_to_idx)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

best_val_accuracy = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_dataloader:
        images = images.to(device=device)
        labels = labels.to(device=device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)

    model.eval()

    correct = 0
    total = 0

    ng_total = 0
    ng_correct = 0

    ok_total = 0
    ok_correct = 0

    with torch.no_grad():

        for images, labels in val_dataloader:
            images = images.to(device=device)
            labels = labels.to(device=device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels, predictions):

                if label.item() == 0:
                    ng_total += 1

                    if prediction.item() == 0:
                        ng_correct += 1

                else:
                    ok_total += 1

                    if prediction.item() == 1:
                        ok_correct += 1

    val_accuracy = correct / total * 100

    ng_accuracy = ng_correct / ng_total * 100
    ok_accuracy = ok_correct / ok_total * 100

    print(
        f"Эпоха {epoch + 1} | "
        f"Loss: {average_loss} | "
        f"Validation: {val_accuracy:.2f}% | "
        f"NG: {ng_accuracy:.2f}% | "
        f"OK: {ok_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "./models/defect_cnn_finetuned.pth"
        )

        print("Сохранена лучшая модель")

print("Лучший Validation:", best_val_accuracy)