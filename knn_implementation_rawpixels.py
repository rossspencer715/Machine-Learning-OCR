import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import zero_pad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
import features
import pylab 

# random seed
M = 3

# class data
#alpha_data = np.load('ClassData.npy')
#labels = np.load('ClassLabels.npy')

# Zare's Data
alpha_data = np.load('TrainImages.npy')
labels = np.load('TrainY.npy')

alpha_data = zero_pad(alpha_data)
a_shape = alpha_data.shape

#reshape the data into a single array
alpha_data = alpha_data[:].reshape(-1, a_shape[1]*a_shape[2])
a_shape = alpha_data.shape

#split the data
X_train, X_valid, label_train, label_valid = train_test_split(alpha_data, labels, test_size=0.3, random_state=M)

label_valid = np.reshape(label_valid, (label_valid.shape[0]))
label_train = np.reshape(label_train, (label_train.shape[0]))
acc = []
macro = []
micro = []
arr = [1,2,3,4,5,6,7,10,15,20,25,40,50,60]
for k in arr:

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, label_train)
    result = knn.predict(X_valid)

    matches = result==label_valid
    correct = np.count_nonzero(matches)
    accuracy = correct/result.size
    print(f'Accuracy: {accuracy*100:.2f}%')
    acc.append(accuracy*100)

    #post-processing, make the non-1s-and-2s class "other"
    labels_true = label_valid
    for i in range(len(label_valid)):
        if label_valid[i] > 2:
            labels_true[i] = -1

    labels_pred = result
    for i in range(len(result)):
        if labels_pred[i] > 2:
            labels_pred[i] = -1

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong

    macro_precision = precision_score(labels_true, labels_pred, average='macro')
    print(f'Macro Precision: {macro_precision*100:.2f}%')
    macro.append(macro_precision*100)

    micro_precision = precision_score(labels_true, labels_pred, average='micro')
    print(f'Micro Precision: {micro_precision*100:.2f}%')
    micro.append(micro_precision*100)

plt.close('all')
plt.plot(arr, acc, 'co', label='Overall Accuracy')
plt.plot(arr, macro, 'mo', label='Macro Precision')
plt.plot(arr, micro, 'o', color='orange', label='Micro Precision')
pylab.legend(loc='upper right')
plt.xlabel('Number of Neighbors')
plt.ylabel('Percentage')
plt.show()