import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
import matplotlib.pyplot as plt

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model_saved_as = 'model_scripted.pt'

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Display class distribution
def class_distribution():

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    
    class_counts = torch.bincount(train_dataset.targets)
    for i, class_name in enumerate(class_names):
        print(f'{class_name}: {class_counts[i]} samples')

#show classes as images
def show_classes():
    labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

#train the model if needed
def train_model(num_epochs):

    model = FashionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training started.....")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

    print('Training complete.')

    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(f'{model_saved_as}') # Save

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (correct / total) * 100
    return accuracy

# Define a function to load and evaluate the model on image data
def evaluate_image(model, image_path):
    model.eval()
    with Image.open(image_path) as image:
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        image = transform(image).unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
    

def main():

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #show classes and their distribution for FashionMNIST Dataset
    class_distribution()
    show_classes()

    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('eval_data_folder', type=str, help='Path to the folder containing the evaluation dataset')
    args = parser.parse_args()

    additional_insights = []

    eval_data_folder = args.eval_data_folder

    # class_distribution()
    # show_classes()

    model = FashionCNN()
    if not os.path.exists(f'{model_saved_as}'):
        print("No trained model found.")
        num_epochs = int(input("Enter the number of epochs you want to train the model: "))
        train_model(num_epochs)
    model = torch.load(f'{model_saved_as}')

    # Check if a GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(eval_data_folder) or not os.listdir(eval_data_folder):
        print("Error: Evaluation folder does not exist or is empty.")
        sys.exit(1)

    if os.path.isdir(eval_data_folder):

        #FashionMNIST Dataset
        if (os.path.exists(os.path.join(eval_data_folder, 'FashionMNIST'))):
            print("FashionMNIST Dataset\n")

            def is_folder_empty(folder_path):
                return not any(os.listdir(folder_path))
            
            if is_folder_empty(os.path.join(eval_data_folder, 'FashionMNIST')):
                additional_insights.append("Folder named FashionMNIST found but empty")

            try:
                transform = transforms.Compose([transforms.ToTensor()])
                eval_dataset = datasets.FashionMNIST(root=eval_data_folder, train=False, transform=transform)
                eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

                # Evaluate the model
                accuracy = evaluate_model(model, eval_loader, device)

                with open("output.txt", "w") as output_file:
                    output_file.write("Model Architecture Summary:\n")
                    output_file.write(str(model) + "\n\n")
                    output_file.write(f"Evaluation Metric (Accuracy): {accuracy:.2f}%\n")

                    output_file.write("Additional Insights or Observations:\n")
                    for num, observabations in enumerate(additional_insights):
                        output_file.write(str(f"    {num}. {observabations}"))

            except Exception as e:
                print(e)
                with open("output.txt", "w") as output_file:
                    output_file.write("Model Architecture Summary:\n")
                    output_file.write("NA" + "\n\n")
                    output_file.write(f"Evaluation Metric (Accuracy): NULL\n\n")

                    output_file.write("Additional Insights or Observations:\n")
                    for num, observabations in enumerate(additional_insights):
                        output_file.write(str(f"    {num}. {observabations}"))

            print(set(additional_insights))

        #Image Folder 
        elif [f for f in os.listdir(eval_data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]:
            print("Image Folder\n")
            
            try:
                image_files = [f for f in os.listdir(eval_data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                results = []
                for image_file in image_files:
                    image_path = os.path.join(eval_data_folder, image_file)
                    try:
                        predicted_label = evaluate_image(model, image_path)
                        results.append({'Image': image_file, 'Predicted Label': predicted_label})
                    except:
                        image_size = Image.open(image_path)
                        width, height = image_size.size
                        additional_insights.append(f"Incompitable Image shape (width x height): {width} x {height}")
                        additional_insights = set(additional_insights)

                with open("output.txt", "w") as output_file:
                    output_file.write("Model Architecture Summary:\n\n")
                    output_file.write(str(model) + "\n\n\n")
                    output_file.write("Predictions from images: \n\n")
                    for result in results:
                        output_file.write(f"{result}\n")
                    
                    output_file.write("\n\nAdditional Insights or Observations:\n")
                    for num, observabations in enumerate(additional_insights):
                        output_file.write(str(f"    {num}. {observabations}"))

            except Exception as e:
                print(e)

            print(additional_insights)

        else:
            print("Unsupported data source type. Please provide an image folder or Fashion MNIST data folder.")
            exit(1)


if __name__ == "__main__":
    main()
