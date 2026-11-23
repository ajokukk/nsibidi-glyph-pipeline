# =====================================================================
# STEP 1: PRETRAIN RESNET-18 ON ADINKRA DATASET
# =====================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Step 1 running on device: {device}")

# 1. Path to your Adinkra dataset folder on Drive
ADINKRA_PATH = '/content/drive/MyDrive/Adinkra_Dataset'

# Image Preprocessing Transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Dataset Loader for Adinkra
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = [], []

        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Initialize DataLoader
adinkra_dataset = ImageFolderDataset(ADINKRA_PATH, transform=transform)
adinkra_loader = DataLoader(adinkra_dataset, batch_size=32, shuffle=True)
num_classes = len(adinkra_dataset.class_names)

# Initialize ResNet-18 Model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes) # Set head to Adinkra class count
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Pretraining Loop (15 Epochs)
print(f"Training ResNet-18 on {num_classes} Adinkra classes...")
model.train()
for epoch in range(15):
    running_loss = 0.0
    for imgs, labels in adinkra_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/15 - Loss: {running_loss/len(adinkra_loader):.4f}")

# Save Pretrained Weights to Drive
SAVE_PATH = '/content/drive/MyDrive/adinkra_resnet18.pth'
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Adinkra Pretrained Weights saved successfully to: {SAVE_PATH}")
