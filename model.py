from PIL import Image
import torch
import torch.nn as nn

# import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import TensorDataset
import os


healthy_circle_dir = "data/Healthy/Circle"
patient_circle_dir = "data/Patient/Circle"

# healthy - 0, parkinson's - 1
healthy_circle = os.listdir(healthy_circle_dir)  # list of strings
patient_circle = os.listdir(patient_circle_dir)

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
train_healthy_imgs = load_imgs(healthy_circle_dir, train_healthy)
test_healthy_imgs = load_imgs(healthy_circle_dir, test_healthy)
train_patient_imgs = load_imgs(patient_circle_dir, train_patient)
test_patient_imgs = load_imgs(patient_circle_dir, test_patient)

train_circle_imgs = torch.cat((train_healthy_imgs, train_patient_imgs), dim=0)
# print(train_circle_imgs.shape)  # tensor with size (n, 3, 224, 224)
train_circle_labels = torch.tensor(
    [0] * len(train_healthy_imgs) + [1] * len(train_patient_imgs)
)

test_circle_imgs = torch.cat((test_healthy_imgs, test_patient_imgs), dim=0)
test_circle_labels = torch.tensor(
    [0] * len(test_healthy_imgs) + [1] * len(test_patient_imgs)
)

train_set = TensorDataset(train_circle_imgs, train_circle_labels)
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True)

test_set = TensorDataset(test_circle_imgs, test_circle_labels)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True)


# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # in_channel, out_channel, ker_size
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)  # ker_size, stride
        self.fc1 = nn.Linear(21632, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


cnn = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)


def compute_accuracy(model, data_loader):
    # Set the model to evaluation mode
    model.eval()

    total = 0
    correct = 0

    for inputs, labels in data_loader:
        outputs = model(inputs)

        # Compute the predicted labels
        _, predicted = torch.max(outputs, 1)

        # print(predicted)

        # Evaluate the accuracy of the model
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Compute the final accuracy
    accuracy = correct / total
    return accuracy


# Training phase

num_epochs = 40

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Clear the gradients
        optimizer.zero_grad()

        # Pass the batch of images through the CNN
        outputs = cnn(images)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Compute the gradients
        loss.backward()

        # Update the weights
        optimizer.step()

    # Print some information about the training progress
    accuracy = compute_accuracy(cnn, train_loader)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%"
    )

# Testing our model

accuracy = compute_accuracy(cnn, test_loader)
print(f"Test accuracy: {accuracy * 100:.2f}%")
