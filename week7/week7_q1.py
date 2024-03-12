import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import PIL
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

# Custom dataset class
class CatDogDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):                                                 
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, label = self.file_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

# Define the neural network architecture
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Adjust output to 2 classes (dogs and cats)

    def forward(self, x):
        return self.model(x)

# Load data
file_paths = []
for class_name in ["cats", "dogs"]:
    for img_path in glob.glob(f"./cats_and_dogs_filtered/train/{class_name}/*.jpg"):
        file_paths.append((img_path, 0 if class_name == "cats" else 1))

# Split data into train and validation sets
train_paths, val_paths = train_test_split(file_paths, test_size=0.2, random_state=42)

# Define transformations
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = CatDogDataset(train_paths, transform=preprocess)
val_dataset = CatDogDataset(val_paths, transform=preprocess)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = CatDogClassifier()

# Define criterion and optimizer with L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Weight decay for L2 regularization

# Training loop
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Train the model
train_model(model, train_dataloader, val_dataloader, criterion, optimizer)
