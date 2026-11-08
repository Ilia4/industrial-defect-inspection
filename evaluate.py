import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DefectCNN

device = (
    torch.accelerator.current_accelerator().type if
    torch.accelerator.is_available() else
    "cpu"
)

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    "./HEAD/Test",
    transform=test_transforms
)

test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

model = DefectCNN()

model.load_state_dict(
    torch.load(
        "./models/defect_cnn.pth",
        map_location=device
    )
)

model = model.to(device=device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:
        images = images.to(device=device)
        labels = labels.to(device=device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

correct = 0

ng = 0
ok = 0

ng_correct = 0
ok_correct = 0

ng_false = 0
ok_false = 0

for label, prediction in zip(all_labels, all_predictions):

    if label == prediction:
        correct += 1

    if label == 0:
        ng += 1

    if label == 1:
        ok += 1

    if label == 0 and prediction == 0:
        ng_correct += 1

    if label == 1 and prediction == 1:
        ok_correct += 1

    if label == 0 and prediction == 1:
        ng_false += 1

    if label == 1 and prediction == 0:
        ok_false += 1

accuracy = correct / len(all_predictions) * 100

recall_ng_percent = ng_correct / ng * 100
recall_ok_percent = ok_correct / ok * 100

precision_ng_percent = (ng_correct / (ng_correct + ok_false)) * 100

precision_ng = precision_ng_percent / 100
recall_ng = recall_ng_percent / 100
f1_ng = (2 * ((precision_ng * recall_ng)/(precision_ng + recall_ng))) * 100 



print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision NG: {precision_ng_percent:.2f}%")
print(f"Recall NG: {recall_ng_percent:.2f}%")
print(f"F1: {f1_ng:.2f}%")
print(f"Recall OK: {recall_ok_percent:.2f}%")

confusion_matrix = [
    [ng_correct, ng_false],
    [ok_false, ok_correct]
]

plt.imshow(confusion_matrix)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.xticks([0, 1], ["NG", "OK"])
plt.yticks([0, 1], ["NG", "OK"])

for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix[i])):
        plt.text(j, i, confusion_matrix[i][j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("./assets/confusion_matrix.png", dpi=150)
plt.show()