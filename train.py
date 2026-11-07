import torch
import torch.nn as nn
import torch.optim as optim
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DefectCNN

EPOCHS = 5

os.makedirs("./models", exist_ok=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    "./HEAD/Train",
    transform=train_transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

model = DefectCNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(EPOCHS):
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

    print(
        f"Эпоха {epoch + 1} | "
        f"Loss: {average_loss}"
    )

torch.save(model.state_dict(), "./models/defect_cnn.pth")
print("Модель сохранена: ./models/defect_cnn.pth")