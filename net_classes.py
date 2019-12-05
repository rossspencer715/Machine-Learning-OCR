import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
import numpy as np
from preprocess import zero_pad
from sklearn.model_selection import train_test_split


# credit: this work was based off of https://medium.com/@athul929/hand-written-digit-classifier-in-pytorch-42a53e92b63e
class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.layer1(x)
        output = self.relu(x1)
        output = self.layer4(output)
        return output


class SimpleCNN(nn.Module):
    # Our batch shape for input x is (1, 66, 143)

    def __init__(self):
        super(SimpleCNN, self, AB=False).__init__()
        # Input channels = 1, output channels = 30, and each is the size of the images
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=30, kernel_size=5)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # we will just always pad the data to this size
        self.fc1 = torch.nn.Linear(15 * 15 * 34, 128)
        self.fc2 = torch.nn.Linear(128, 50)
        # # 64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(50, 9)
        if AB:
            self.fc3 = torch.nn.Linear(50, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 15 * 15 * 34)
        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def load_data(zare_data=True, AB=True):
    if zare_data:
        data = np.load('TrainImages.npy')
        labels = np.load('TrainY.npy')
    else:
        data = np.load('ClassData.npy')
        labels = np.load('ClassLabels.npy')

    data = zero_pad(data)
    labels = np.reshape(labels, (labels.shape[0]))

    if AB:
        labels[labels > 2] = 0

    X_train, X_valid, label_train, label_valid = train_test_split(data, labels, test_size=0.3, random_state=3)

    train_data = torch.from_numpy(X_train).float()
    train_labels = torch.from_numpy(label_train).long()

    valid_data = torch.from_numpy(X_valid).float()
    valid_labels = torch.from_numpy(label_valid).long()

    full_data = torch.from_numpy(data).float()
    full_labels = torch.from_numpy(labels).long()

    tensor_data = torch.stack([torch.Tensor(i) for i in train_data[:]])
    tensor_labels = torch.stack([torch.tensor(i) for i in train_labels[:]])

    val_tensor_data = torch.stack([torch.Tensor(i) for i in valid_data[:]])
    val_tensor_labels = torch.stack([torch.tensor(i) for i in valid_labels[:]])

    full_tensor_data = torch.stack([torch.Tensor(i) for i in full_data[:]])
    full_tensor_labels = torch.stack([torch.tensor(i) for i in full_labels[:]])

    my_dataset = utils.TensorDataset(tensor_data, tensor_labels)
    my_dataloader = utils.DataLoader(my_dataset)

    val_my_dataset = utils.TensorDataset(val_tensor_data, val_tensor_labels)
    val_my_dataloader = utils.DataLoader(val_my_dataset)

    full_my_dataset = utils.TensorDataset(full_tensor_data, full_tensor_labels)
    full_my_dataloader = utils.DataLoader(full_my_dataset)
    return my_dataloader, val_my_dataloader, train_data, full_my_dataloader, labels, labels.shape[0]


def load_data_from_file(indata_fp):
    # zero pad the data appropriately
    data = zero_pad(np.load(indata_fp))
    # convert to Tensor dataset
    full_data = torch.from_numpy(data).float()
    tensor_data = torch.stack([torch.Tensor(i) for i in full_data[:]])
    full_my_dataset = utils.TensorDataset(tensor_data)
    # return the data loader
    return utils.DataLoader(full_my_dataset)


def eval_model(model, val_loader, CNN=True):
    prediction_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if not CNN:
                images = images.reshape(-1, images.shape[1]*images.shape[2])
            else:
                images = images[np.newaxis, :]
            # have the model guess at the classification
            guess = model(images)
            # take the prediction
            prediction_labels.insert(i, torch.argmax(guess.data,1))
    return np.array(prediction_labels)

# if the data does not have the labels with it
def eval_model_no_labels(model, val_loader, CNN=True):
    #torch.set_printoptions(threshold=1e10)
    prediction_labels = []
    with torch.no_grad():
        for i, images in enumerate(val_loader):
            if not CNN:
                images = images[0].reshape(-1, images[0].shape[1]*images[0].shape[2])
            else:
                images = images[0].unsqueeze(0)#[np.newaxis, 0]
            # have the model guess at the classification
            guess = model(images)
            # take the prediction
            prediction_labels.insert(i, torch.argmax(guess.data,1))
    return np.array(prediction_labels)


