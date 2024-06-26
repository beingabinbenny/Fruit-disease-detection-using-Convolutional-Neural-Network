# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import seaborn as sns

# Required magic to display matplotlib plots in notebooks
#%matplotlib inline

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import shutil
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



##########################################################################################################################

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import os
from PIL import Image
import shutil





# Set the path to the SenMangoFruitDDS dataset
training_folder_name = '/Users/abinbenny/Documents/adithya project/fruit/MangoFruitDDS/SenMangoFruitDDS_original'
# Define the classes based on the diseases and "Healthy" category
classes = ['Alternaria', 'Anthracnose', 'Black_Mould_Rot', 'Stem_and_Rot', 'Healthy']

# Set image size for the CNN model
img_size = (128, 128)

# Create a folder for resized images
train_folder = '/Users/abinbenny/Documents/adithya project/fruit/MangoFruitDDS/Mango'



##########################################################################################################################

 
# Function to resize image
def resize_image(src_image, size=(128, 128), bg_color="white"):
    src_image.thumbnail(size, Image.LANCZOS)
    new_image = Image.new("RGB", size, bg_color)
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    return new_image

# Create resized copies of all the images
size = (128, 128)

if os.path.exists(train_folder):
    shutil.rmtree(train_folder)

for root, folders, files in os.walk(training_folder_name):
    for sub_folder in folders:
        save_folder = os.path.join(train_folder, sub_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_names = os.listdir(os.path.join(root, sub_folder))
        for file_name in file_names:
            file_path = os.path.join(root, sub_folder, file_name)
            image = Image.open(file_path)
            resized_image = resize_image(image, size)
            save_as = os.path.join(save_folder, file_name)
            resized_image.save(save_as)




##########################################################################################################################




# Data augmentation and preprocessing
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset
full_dataset = torchvision.datasets.ImageFolder(
    root=train_folder,
    transform=data_transform
)

# Split into training and testing datasets
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)





##########################################################################################################################


'''
# Define the CNN model
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(self.drop(x), training=self.training)
        x = x.view(-1, 32 * 32 * 24)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

# Instantiate the model
model = Net(num_classes=len(classes))

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_criteria = nn.CrossEntropyLoss()

'''


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# Instantiate the model
model = Net(num_classes=len(classes))

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_criteria = nn.CrossEntropyLoss()

##########################################################################################################################




# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_criteria(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {:.2f}%'.format(avg_loss, accuracy * 100))











##########################################################################################################################



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criteria(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = train_loss / len(train_loader)
    print('Epoch {}: Train set: Average loss: {:.6f}'.format(epoch, avg_loss))
    return avg_loss

