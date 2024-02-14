# Intotech
## Opis projektu
Projekt Rozpoznawania Emocji wykorzystuje model głębokiego uczenia oparty na architekturze ResNet18 do identyfikacji emocji na zdjęciach twarzy. Celem jest klasyfikacja obrazów do jednej z siedmiu kategorii emocji: szczęście, smutek, zaskoczenie, strach, gniew, pogarda, neutralność.

## Wymagania
Aby uruchomić projekt, potrzebne są następujące biblioteki:

Python 3.8+
PyTorch 1.9+
torchvision 0.10+
PIL (Python Imaging Library)
NumPy
Zaleca się użycie środowiska wirtualnego Pythona dla lepszego zarządzania zależnościami.

## Przygotowanie danych
Dane powinny być zorganizowane w następującej strukturze katalogów:

```bash

data/
├── train/
│   ├── szczęście/
│   ├── smutek/
│   ├── zaskoczenie/
│   └── ...
└── test/
    ├── szczęście/
    ├── smutek/
    ├── zaskoczenie/
    └── ...
```
Każdy podkatalog zawiera obrazy należące do jednej z kategorii emocji.

## Transformacje
Dane są przetwarzane przy użyciu następujących transformacji:

Przeskalowanie i przycięcie do 224x224 pikseli
Losowe lustrzane odbicie (tylko dane treningowe)
Normalizacja z użyciem średnich i odchyleń standardowych dla ImageNet
## Trenowanie modelu
Aby wytrenować model, uruchom skrypt train.py, który automatycznie przetrenuje model na danych treningowych. Możesz dostosować parametry treningu, takie jak liczba epok czy rozmiar batcha, wewnątrz skryptu.

## Ewaluacja modelu
Wytrenowany model można ocenić na zbiorze testowym za pomocą skryptu evaluate.py. Skrypt ten załaduje wytrenowany model i obliczy jego dokładność na danych testowych.

Przewidywanie emocji
Aby przewidzieć emocje na nowym obrazie, użyj funkcji predict_emotion z predict.py, przekazując ścieżkę do obrazu. Funkcja ta zwróci przewidywaną klasę emocji.

## Przykład użycia:

```python

image_path = 'ścieżka/do/obrazu.jpg'
predicted_emotion = predict_emotion(model, image_path, transform)
print(f'Przewidziana emocja: {predicted_emotion}')

 
