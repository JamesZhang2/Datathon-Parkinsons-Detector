from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import TensorDataset
import os


# healthy - 0, parkinson's - 1
healthy_circle = os.listdir("data/Healthy/Circle")  # list of strings
patient_circle = os.listdir("data/Patient/Circle")

circle_imgs = healthy_circle
train_healthy, test_healthy = train_test_split(healthy_circle, test_size=0.2)
train_patient, test_patient = train_test_split(patient_circle, test_size=0.2)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Normalize the tensor values
    ]
)


def load_imgs(dir, file_names):
    ret = []
    for file_name in file_names:
        image_path = os.path.join(dir, file_name)
        image = Image.open(image_path)

        image = transform(image)  # torch.Tensor with shape (3, 224, 224)
        # print(image.shape)
        ret.append(image)
    ret_tensor = torch.stack(ret, dim=0)
    # print(ret_tensor.size())
    return ret_tensor  # torch.Tensor with shape (n, 3, 224, 224)


# 4D tensors
train_healthy_imgs = load_imgs("data/Healthy/Circle", train_healthy)
test_healthy_imgs = load_imgs("data/Healthy/Circle", test_healthy)
train_patient_imgs = load_imgs("data/Patient/Circle", train_patient)
test_patient_imgs = load_imgs("data/Patient/Circle", test_patient)

train_circle_imgs = torch.cat((train_healthy_imgs, train_patient_imgs), dim=0)
# print(train_circle_imgs.shape)  # tensor with size (n, 3, 224, 224)
train_circle_labels = torch.tensor(
    [0] * len(train_healthy_imgs) + [1] * len(train_patient_imgs)
)

dataset = TensorDataset(train_circle_imgs, train_circle_labels)
# print(type(dataset))

dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)


# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
