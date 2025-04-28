import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --- Definir modelo igual que antes ---
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Configuraciones ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 4  # Número de clases que entrenaste (ajústalo si es necesario)

# --- Cargar modelo ---
model = CNN(num_classes)
model.load_state_dict(torch.load('modelo_basura.pth', map_location=device))
model = model.to(device)
model.eval()

# --- Transformación para imagen ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Función para predecir una imagen ---
def predecir_imagen(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    clases = ['metal', 'orgánico', 'papel', 'plástico']  # ajusta si tienes otros nombres
    print(f"La imagen {os.path.basename(image_path)} es clasificada como: {clases[predicted.item()]}")

# --- Aquí pones la ruta de prueba ---
ruta_imagen = r'C:\Users\Marisella Davalos\Desktop\medioambiente\pruebas\prueba10.jpg'
predecir_imagen(ruta_imagen)
