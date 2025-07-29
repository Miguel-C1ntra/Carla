import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import time

class CarlaDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_id = str(self.data.iloc[idx, 0])
        img_path = os.path.join(self.image_dir, f"{frame_id}_front.png")  # usa câmera frontal
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        steer = float(self.data.iloc[idx, 1])
        return image, torch.tensor([steer], dtype=torch.float32)

class SteeringCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            flat = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def train_model():
    dataset = CarlaDataset("dataset/labels.csv", "dataset/images")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SteeringCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("📚 Treinando modelo...")
    for epoch in range(5):
        total_loss = 0.0
        for images, steers in loader:
            images, steers = images.to(device), steers.to(device)
            pred = model(images)
            loss = criterion(pred, steers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss médio: {total_loss / len(loader):.5f}")

    torch.save(model.state_dict(), "steering_model.pth")
    print("✅ Modelo salvo como 'steering_model.pth'")

if __name__ == "__main__":
    train_model()




