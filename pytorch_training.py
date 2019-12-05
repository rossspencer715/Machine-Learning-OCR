import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from net_classes import *
from sklearn.metrics import precision_score, accuracy_score
import time

def trainCNNNet(net, train_loader, val_loader, n_epochs, learning_rate):
    """This will train the CNN, saves the model each epoch, and writes to a
      statistics array for graphing"""
    print(f"Beginning training...\nEpochs: {n_epochs}\nLearning rate: {learning_rate}\nBatch Size: {len(train_loader)}")
    # declare the stats array
    stats_array = np.zeros((2, n_epochs))
    # declare the number of batches
    samples = len(train_loader)
    # define the labels
    true_labels = np.array([labels for (images, labels) in val_loader])
    # define loss function
    loss = torch.nn.CrossEntropyLoss()
    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # Loop for the number of epochs and track the amount of time that the whole training takes
    train_start_time = time.time()
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = samples // 10
        start_time = time.time()
        total_train_loss = 0
        max_loss = 0
        for i, (inputs, labels) in enumerate(train_loader, 0):

            # Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            # Forward pass, backward pass, optimize
            outputs = net(inputs[np.newaxis,:])
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data[0].item()
            total_train_loss += loss_size.data[0].item()

            if (max_loss < running_loss / print_every):
                max_loss = running_loss / print_every

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / samples), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            # Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            # Pass the data through the network
            val_outputs = net(inputs[np.newaxis, :])
            # determine the amount of loss
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
        print(f'Validation loss = {total_val_loss / len(val_loader):.2f}')

        # validate the data

        predicted = eval_model(net, val_loader, CNN=True)
        print(f'Current Accuracy = {accuracy_score(true_labels, predicted)*100:.2f}%')
        stats_array[0, epoch] = accuracy_score(true_labels, predicted)*100
        stats_array[1, epoch] = max_loss
        torch.save(model, f'./models/temp_models/CNN_classifier_epoch_{epoch}.pt')
    print("Training finished, took {:.2f}s".format(time.time() - train_start_time))
    return stats_array


def train_NN(net, train_loader, val_loader, n_epochs, learning_rate):
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    total_batch = len(train_loader)
    stats = np.zeros((2,n_epochs))
    for epoch in range(n_epochs):
        this = 0
        max_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_lay_num)
            out = net(images)
            loss = lossFunction(out,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f'Epoch {epoch+1}: Step [{i+1}/{total_batch}], Loss: {loss.item():.3f}')
                this = loss.item();
            if (max_loss < this):
                max_loss = this
        true_labels = np.array([labels for images,labels in val_loader])
        predicted = eval_model(net, val_loader, CNN=False)
        # 0 index will have the model accuracy over each epoch
        stats[0, epoch] = accuracy_score(true_labels, predicted)*100
        # 1 index will have the model loss over each epoch
        stats[1, epoch] = max_loss
        print("Validation loss = ", max_loss)
        print(f'Current Accuracy = {accuracy_score(true_labels, predicted)*100:.2f}%')
    return stats


my_dataloader, val_my_dataloader, train_data, full_my_dataloader, true_labels, label_size = load_data(
    zare_data=True, AB=True)

# the input layer has the same number of pixels as are in the images
input_lay_num = train_data.shape[1]*train_data.shape[2]
# this is just a guess and check sort of deal
hidden_lay_num = 500

CNN = False
AB = True

# number of letters that we are processing
if AB:
    output_size = 3
else:
    output_size = 9
lr = 0.001


if CNN:
    model = SimpleCNN(AB=AB)
    my_dataloader, val_my_dataloader, train_data, full_my_dataloader, true_labels, label_size = load_data(
        zare_data=True, AB=AB)
    # this will essentially create a "pretrained model" on the class data
    statistics = trainCNNNet(model, my_dataloader, val_my_dataloader, 30, learning_rate=lr)
    torch.save(model, './models/CNN_class_all_letters_classifier_30_epochs.pt')
    np.save('./data/CNN_class_all_letters_classifier_30_epochs.npy', statistics)

#place an else statement here to use the CNN tag to choose between a standard neural net training and a CNN training
    model = NeuralNet(input_lay_num, hidden_lay_num,  output_size)
    my_dataloader, val_my_dataloader, train_data, full_my_dataloader, true_labels, label_size = load_data(
        zare_data=True, AB=AB)
    statistics = train_NN(model, my_dataloader, val_my_dataloader, 30, learning_rate=lr)
    torch.save(model, './models/NN_class_all_letters_classifier_30_epochs.pt')
    np.save('./data/NN_class_all_letters_classifier_30_epochs.npy', statistics)


