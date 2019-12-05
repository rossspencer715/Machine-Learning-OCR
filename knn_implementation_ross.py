import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import zero_pad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
import features

# random seed
M = 3

# class data
alpha_data = np.load('ClassData.npy')
labels = np.load('ClassLabels.npy')

# Zare's Data
alpha_data = np.load('TrainImages.npy')
labels = np.load('TrainY.npy')

alpha_data = zero_pad(alpha_data)
a_shape = alpha_data.shape

# reshape the data into a single array
#alpha_data = alpha_data[:].reshape(-1, a_shape[1]*a_shape[2])
#a_shape = alpha_data.shape

#split the data
X_train, X_valid, label_train, label_valid = train_test_split(alpha_data, labels, test_size=0.3, random_state=M)

label_valid = np.reshape(label_valid, (label_valid.shape[0]))
label_train = np.reshape(label_train, (label_train.shape[0]))
# Initiate kNN, train the data, then test it with test data for k=1
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, label_train)
result = knn.predict(X_valid)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==label_valid
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)

labels_true = label_valid
for i in range(len(label_valid)):
    if label_valid[i] > 2:
        labels_true[i] = -1

labels_pred = result
for i in range(len(result)):
    if labels_pred[i] > 2:
        labels_pred[i] = -1


precision = precision_score(labels_true, labels_pred, average='macro')
print(precision)


precision = precision_score(labels_true, labels_pred, average='micro')
print(precision)


#def bounding_box_slower(img):
#    a = np.where(img != 0)
#    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
#    return bbox

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymax, ymin = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return ymax, ymin, xmin, xmax

def mean(img):
    return np.mean(img, dtype=np.float64)

print(bounding_box(alpha_data[0]))
img_show = plt.imshow(alpha_data[150])
%timeit bbox2(alpha_data[0])
mean(alpha_data[0])
%timeit mean(alpha_data[0])

from skimage.measure import regionprops
labeled_foreground = (alpha_data[0] > 0).astype(int)
properties = regionprops(labeled_foreground, alpha_data[0])
center_of_mass = properties[0].centroid
weighted_center_of_mass = properties[0].weighted_centroid

def centroid(img):
    labeled_foreground = (img > 0).astype(int) ##threshold value := 0
    properties = regionprops(labeled_foreground, img)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid
    return center_of_mass

%timeit centroid(alpha_data[150])

box_center(alpha_data[150])
def char_width(x1, x2):
    return (x1 - x2)

def char_height(y1, y2):
    return (y1 - y2)

np.sum(alpha_data[0])
num_on(alpha_data[0])
avg(alpha_data[150])

img = alpha_data[150]
img_show = plt.imshow(alpha_data[150])
on = np.where(img == 1)
np.min(on[1])
center = centroid(img)
horiz, vert, width, height = box_center(img)
on0 = [y - center[0] for y in on[0]]
on1 = [x - center[1] for x in on[1]]
avg_x = (np.sum(on1)/width)
avg_y = (np.sum(on0)/height)
# feature 6, feature 7
avg_x, avg_y

print(avg(img))
