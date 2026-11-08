import torch

from torchvision import transforms
from PIL import Image

from model import DefectCNN

device = (
    torch.accelerator.current_accelerator().type if
    torch.accelerator.is_available() else "cpu"
)

model = DefectCNN()

model.load_state_dict(
    torch.load(
        "./models/defect_cnn.pth",
        map_location=device
    )
)
model = model.to(device=device)

model.eval()

image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

image = Image.open("./HEAD/Test/OK/OK-12.BMP").convert("RGB")
image_tensor = image_transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device=device)

with torch.no_grad():
    outputs = model(image_tensor)
    prediction = torch.argmax(outputs, dim=1)

    answer = prediction.item()

    conf = torch.softmax(input=outputs, dim=1)

    confidence = conf[0][answer].item() * 100

    if answer == 0:
        print("Predict: NG")
    else:
        print("Predict: OK")

    print(f"Confidence: {confidence:.2f}%")
