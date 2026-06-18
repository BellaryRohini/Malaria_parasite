import os
import random

# Imports for Managing Datasets
import numpy as np
import pandas as pd

# Imports for Data Visualization
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
# Directory for Parasitized cell images
parasitize_images_dir = 'cell_images/Parasitized'

# Directory for Uninfected cell images
uninfected_images_dir = 'cell_images/Uninfected'
# lets take a look at the number of images of parasitized cells
import os

parasitize_images_dir = 'cell_images/Parasitized'

print("Folder exists:", os.path.exists(parasitize_images_dir))
print("Path:", parasitize_images_dir)

num_parasitized_images = len(os.listdir(parasitize_images_dir))
print(num_parasitized_images)
# lets take a look at the number of images of uninfected cells
import os

uninfected_images_dir = 'cell_images/Uninfected'

num_uninfected_images = len(os.listdir(uninfected_images_dir))
print(num_uninfected_images)
# lets see a variety of parasitized cells
filenames = random.sample(os.listdir('cell_images/Parasitized/'), 25)

# here we will see 25 images of Parasitized cell images
plt.figure(figsize=(15, 15))  # figure size

for i in range(1, len(filenames)):
    row = i
    image = imread('cell_images/Parasitized/' + filenames[i])
    plt.subplot(5, 5, row)
    plt.imshow(image)

plt.show()
# lets see a variety of uninfected cells
filenames_ = random.sample(os.listdir('cell_images/Uninfected/'), 26)

# here we will see 25 images of Uninfected cell images
plt.figure(figsize=(15, 15))

for i in range(1, len(filenames)):
    row = i
    image = imread('cell_images/Uninfected/' + '/' + filenames_[i])
    plt.subplot(5, 5, row)
    plt.imshow(image)

plt.show()
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# import for train-test-split
from sklearn.model_selection import train_test_split

# import for One Hot Encoding
from keras.utils import to_categorical

# importing libraries for Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

# importing libraries for evaluating the model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data = []
labels = []
Parasitized = os.listdir('cell_images/Parasitized/')

for a in Parasitized:

    try:
        image = cv2.imread('cell_images/Parasitized/' + a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(0)

    except AttributeError:
        print("")
Uninfected = os.listdir('cell_images/Uninfected/')

for b in Uninfected:

    try:
        image = cv2.imread('cell_images/Uninfected/' + b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(1)

    except AttributeError:
        print("")
data = np.array(data)
labels = np.array(labels)
np.save('Data' , data)
np.save('Labels' , labels)
print('Cells : {} and labels : {}'.format(data.shape , labels.shape))
plt.figure(figsize = (7 , 7))
plt.imshow(data[100])
plt.title('Parasitized Cell')
labels[100]
# lets take a look at an Uninfected cell
plt.figure(figsize = (7 , 7))
plt.imshow(data[15000])
plt.title('Uninfected Cell')
labels[15000]
n = np.arange(data.shape[0])
np.random.shuffle(n)
data = data[n]
labels = labels[n]
plt.figure(figsize = (7 , 7))
plt.imshow(data[10000])
labels[10000]
# 1 is uninfected and 0 is parasitized
X_train, X_valid, y_train, y_valid = train_test_split(data, labels, test_size = 0.2, random_state = 0)

print('Train data shape {} ,Test data shape {} '.format(X_train.shape, X_valid.shape))
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
# Defining Model
classifier = Sequential()

# CNN layers
classifier.add(Conv2D(32, kernel_size=(3, 3), input_shape = (50, 50, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis = -1))
classifier.add(Dropout(0.5))   # Dropout prevents overfitting

classifier.add(Conv2D(32, kernel_size=(3, 3), input_shape = (50, 50, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis = -1))
classifier.add(Dropout(0.5))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(BatchNormalization(axis = -1))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=2, activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, batch_size=120, epochs=25, verbose=1, validation_data=(X_valid, y_valid))
print("Test_Accuracy: {:.2f}%".format(classifier.evaluate(X_valid, y_valid)[1]*100))
classifier.summary()
y_pred = classifier.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)
y_valid = np.argmax(y_valid, axis=1)
print('Accuracy Score: ', accuracy_score(y_valid, y_pred))
# Plotting the Confusion Matrix
conf = confusion_matrix(y_valid, y_pred)
sns.heatmap(conf, annot=True)
# train and validation Accuracy trends
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = [i for i in range(0, 25)]
plt.plot(epochs, train_accuracy, 'r', label='Train Accuracy')
plt.plot(epochs, val_accuracy, 'g', label='Validation Accuracy')
plt.title('Train vs Validation Accuracy Trends')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# train and validation loss trends
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = [i for i in range(0, 25)]
plt.plot(epochs, train_loss, 'r', label='Train Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Train vs Validation Loss Trends')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
import numpy as np

# Calculate sensitivity for each class
sensitivities = {}
num_classes = conf.shape[0]

for i in range(num_classes):
    tp = conf[i, i]  # True Positives for class i
    fn = conf[i, :].sum() - tp  # False Negatives for class i (sum of all predictions for class i minus TP)
    sensitivity = tp / (tp + fn)  # Sensitivity for class i
    sensitivities[f"Class {i}"] = sensitivity

# Print Sensitivities for all classes
for class_name, sensitivity in sensitivities.items():
    print(f"Sensitivity for {class_name}: {sensitivity * 100:.2f}%")
import numpy as np
from sklearn.metrics import confusion_matrix
# Initialize dictionary to store metrics for each class
metrics = {}
num_classes = conf.shape[0]

# Loop over each class to compute sensitivity, specificity, and F1 score
for i in range(num_classes):
    tp = conf[i, i]  # True Positives for class i
    fn = conf[i, :].sum() - tp  # False Negatives for class i (sum of row minus TP)
    fp = conf[:, i].sum() - tp  # False Positives for class i (sum of column minus TP)
    tn = conf.sum() - (tp + fn + fp)  # True Negatives for class i (total sum minus TP, FN, FP)

    # Sensitivity (Recall) for class i
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Specificity for class i
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Precision for class i
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # F1 Score for class i
    if (precision + sensitivity) != 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0

    # Store metrics in dictionary
    metrics[f"Class {i}"] = {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1 Score': f1_score
    }

# Print metrics for all classes
for class_name, metric_values in metrics.items():
    print(f"\n{class_name}:")
    print(f"Sensitivity: {metric_values['Sensitivity'] * 100:.2f}%")
    print(f"Specificity: {metric_values['Specificity'] * 100:.2f}%")
    print(f"Precision: {metric_values['Precision'] * 100:.2f}%")
    print(f"F1 Score: {metric_values['F1 Score'] * 100:.2f}%")
