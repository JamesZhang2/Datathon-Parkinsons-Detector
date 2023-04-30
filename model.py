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
healthy_meander_dir = "data/Healthy/Meander"
patient_meander_dir = "data/Patient/Meander"
healthy_spiral_dir = "data/Healthy/Spiral"
patient_spiral_dir = "data/Patient/Spiral"


def load_imgs(dir, file_names, transform):
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


class Net(nn.Module):
    # Define the CNN architecture
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # in_channel, out_channel, ker_size
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=2)
        self.pool1 = nn.MaxPool2d(2, 2)  # ker_size, stride
        self.pool2 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool1(nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_accuracy(model, data_loader):
    # Set the model to evaluation mode
    model.eval()

    total = 0
    correct = 0

    for inputs, labels in data_loader:
        # print(inputs.shape)  # (1, 3, 224, 224)
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


class ParkinsonPredictor:
    def __init__(self, num_epochs, retrain=False, save=True):
        if (not retrain) and (os.path.exists("model.pth")):
            print("Loading saved model")
            self.cnn = Net()
            self.cnn.load_state_dict(torch.load("model.pth"))
        else:
            self.cnn = self.train_model(num_epochs, save)

    def train_model(self, num_epochs, save):
        print("Training model")
        healthy_dir = healthy_meander_dir
        patient_dir = patient_meander_dir

        # healthy - 0, parkinson's - 1
        healthy_filenames = os.listdir(healthy_dir)  # list of strings
        patient_filenames = os.listdir(patient_dir)

        train_healthy, test_healthy = train_test_split(healthy_filenames, test_size=0.2)
        train_patient, test_patient = train_test_split(patient_filenames, test_size=0.2)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Normalize the tensor values
            ]
        )

        # Load images as 4D tensors
        train_healthy_imgs = load_imgs(healthy_dir, train_healthy, self.transform)
        test_healthy_imgs = load_imgs(healthy_dir, test_healthy, self.transform)
        train_patient_imgs = load_imgs(patient_dir, train_patient, self.transform)
        test_patient_imgs = load_imgs(patient_dir, test_patient, self.transform)

        train_imgs = torch.cat((train_healthy_imgs, train_patient_imgs), dim=0)
        # print(train_imgs.shape)  # tensor with size (n, 3, 224, 224)
        train_labels = torch.tensor(
            [0] * len(train_healthy_imgs) + [1] * len(train_patient_imgs)
        )

        test_imgs = torch.cat((test_healthy_imgs, test_patient_imgs), dim=0)
        test_labels = torch.tensor(
            [0] * len(test_healthy_imgs) + [1] * len(test_patient_imgs)
        )

        train_set = TensorDataset(train_imgs, train_labels)
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True)

        test_set = TensorDataset(test_imgs, test_labels)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=True)

        self.cnn = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.cnn.parameters(), lr=0.001)

        # Training phase
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                # Clear the gradients
                optimizer.zero_grad()

                # Pass the batch of images through the CNN
                outputs = self.cnn(images)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Compute the gradients
                loss.backward()

                # Update the weights
                optimizer.step()

            # Print some information about the training progress
            accuracy = compute_accuracy(self.cnn, train_loader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%"
            )

        # Testing
        accuracy = compute_accuracy(self.cnn, test_loader)
        print(f"Test accuracy: {accuracy * 100:.2f}%")

        # Save the model
        if save:
            print("Saving model to model.pth")
            torch.save(self.cnn.state_dict(), "model.pth")

    def predict(self, image):
        """Return 0 if healthy, 1 if patient"""
        image_tensor = torch.stack([self.transform(image)], dim=0)
        # print(image_tensor.shape)  # (1, 3, 224, 224)
        self.cnn.eval()
        outputs = self.cnn.forward(image_tensor)

        # Compute the predicted labels
        _, predicted = torch.max(outputs, 1)
        return predicted
