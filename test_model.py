import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from main import load_model
import numpy as np

def detailed_evaluation(model_path, test_loader, device, class_names):
    model = load_model('best_model.pth')  # Załaduj model
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Oblicz dokładność
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Wyświetl raport klasyfikacji
    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=class_names))


