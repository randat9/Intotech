import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Definicja transformacji
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Załadowanie modelu
def load_model(model_path):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)  # Zakładamy, że mamy 7 kategorii emocji
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    return model

model = load_model('best_model.pth')


def predict_emotion(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Dodanie wymiaru batch

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Zdefiniuj swoje etykiety emocji zgodnie z kolejnością treningową
    emotion_labels = ['szczęście', 'smutek', 'zaskoczenie', 'strach', 'gniew', 'pogarda', 'neutralny']
    predicted_emotion = emotion_labels[predicted.item()]

    return predicted_emotion

# Ścieżka do obrazu, który chcesz przetestować
image_path = 'C:/Users/Administrator/Documents/GitHub/Intotech/zaskoczenie.jpg'

# Przewidywanie emocji
emotion = predict_emotion(model, image_path, transform)
print(f'Przewidziana emocja: {emotion}')

