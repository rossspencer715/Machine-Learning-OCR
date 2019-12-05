import numpy as np
import matplotlib.pyplot as plt

zare_data_AB_cnn_30_epochs = np.load('CNN_AB_zare_classifier_30_epochs.npy')
zare_data_AB_rnn_30_epochs = np.load('zare_data_AB_rnn_30_epochs.npy')
class_data_all_cnn_30_epochs = np.load('CNN_class_all_letters_classifier_30_epochs.npy')
class_data_all_rnn_30_epochs = np.load('NN_class_all_letters_classifier_30_epochs.npy')

ax = plt.axes()
plt.title('Validation Accuracy vs. # of Epoch Iterations')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy %')
x_axis = np.arange(1,31)
plt.plot(x_axis, zare_data_AB_cnn_30_epochs[0,:], label='Convolutional NN')
plt.plot(x_axis, zare_data_AB_rnn_30_epochs[0,:], label='Standard NN')
plt.legend()
plt.show()

plt.title('Validation Loss vs. # of Epoch Iterations')
plt.xlabel('Epoch #')
plt.ylabel('Loss Value')
x_axis = np.arange(1,31)
plt.plot(x_axis, zare_data_AB_cnn_30_epochs[1,:], label='Convolutional NN')
plt.plot(x_axis, zare_data_AB_rnn_30_epochs[1,:], label='Standard NN')
plt.legend()
plt.show()

ax = plt.axes()
plt.title('Validation Accuracy vs. # of Epoch Iterations')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy %')
x_axis = np.arange(1,31)
plt.plot(x_axis, class_data_all_cnn_30_epochs[0,:], label='Convolutional NN', color='c')
plt.plot(x_axis, class_data_all_rnn_30_epochs[0,:], label='Standard NN', color='m')
plt.legend()
plt.show()

plt.title('Validation Loss vs. # of Epoch Iterations')
plt.xlabel('Epoch #')
plt.ylabel('Loss Value')
x_axis = np.arange(1,31)
plt.plot(x_axis, class_data_all_cnn_30_epochs[1,:], label='Convolutional NN', color='c')
plt.plot(x_axis, class_data_all_rnn_30_epochs[1,:], label='Standard NN', color='m')
plt.legend()
plt.show()

