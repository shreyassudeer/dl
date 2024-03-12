import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import PIL
import glob
from PIL import Image
from matplotlib import pyplot as plt

# Define Gaussian noise transformation
class Gaussian(object):
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img + torch.normal(self.mean, self.var, img.size())

# Data augmentation transformations
preprocess_augmented = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.RandomHorizontalFlip(),
    T.RandomRotation(45),
    Gaussian(0, 0.15),
])

# Without data augmentation
preprocess_no_augmentation = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, transform=None, str="train"):
        self.imgs_path = "./cats_and_dogs_filtered/" + str + "/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = PIL.Image.open(img_path)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        if self.transform:
            img = self.transform(img)
        return img, class_id

# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Adjust output to 2 classes (dogs and cats)

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Create datasets and dataloaders
dataset_augmented = MyDataset(transform=preprocess_augmented, str="train")
dataset_no_augmentation = MyDataset(transform=preprocess_no_augmentation, str="train")
dataloader_augmented = DataLoader(dataset_augmented, batch_size=32, shuffle=True)
dataloader_no_augmentation = DataLoader(dataset_no_augmentation, batch_size=32, shuffle=True)

# Initialize the model, criterion, and optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train with data augmentation
print("Training with data augmentation:")
train_model(model, dataloader_augmented, criterion, optimizer)

# Train without data augmentation
print("\nTraining without data augmentation:")
train_model(model, dataloader_no_augmentation, criterion, optimizer)
