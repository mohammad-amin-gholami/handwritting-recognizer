import cv2
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


X_train, y_train = read_hoda_dataset(dataset_path='Train 60000.cdb',
                                     images_height=32,
                                     images_width=32,
                                     one_hot=False,
                                     reshape=True)

X_test, y_test = read_hoda_dataset(dataset_path='Test 20000.cdb',
                                   images_height=32,
                                   images_width=32,
                                   one_hot=False,
                                   reshape=True)

X_remaining, y_remaining = read_hoda_dataset(dataset_path='RemainingSamples.cdb',
                              images_height=32,
                              images_width=32,
                              one_hot=False,
                              reshape=True)


# Visualising the three sample

fig = plt.figure(figsize=(16, 3))
fig.add_subplot(1, 3, 1)
plt.title('Y_train[ 0 ] = ' + str(y_train[0]))
plt.imshow(X_train[0].reshape([32, 32]), cmap='gray')

fig.add_subplot(1, 3, 2)
plt.title('Y_test[ 0 ] = ' + str(y_test[0]))
plt.imshow(X_test[0].reshape([32, 32]), cmap='gray')

fig.add_subplot(1, 3, 3)
plt.title('y_remaining[ 0 ] = ' + str(y_remaining[0]))
plt.imshow(X_remaining[0].reshape([32, 32]), cmap='gray')

plt.show()

# flatten
X_train_flattened = X_train.reshape(len(X_train), 32*32)
X_test_flattened = X_test.reshape(len(X_test), 32*32)

print(X_train_flattened.shape)

print(X_test_flattened.shape)

#Fit a model
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(1024,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=10)


model.evaluate(X_test_flattened, y_test)

y_predicted = model.predict(X_test_flattened)
print(y_predicted.shape)
# funny test
# plt.imshow(X_test[6353].reshape([32,32]))
# np.argmax(y_predicted[6353])

# Confusionmatrix with swabord

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Testing model using external sample

original_img = cv2.imread("Test_image/4.jpg")
resized_img = cv2.resize(original_img, (32, 32))

resultNot = cv2.bitwise_not(src=resized_img, dst=resized_img)
(thresh, blackAndWhiteImage) = cv2.threshold(resultNot, 127, 255, cv2.THRESH_BINARY)
_, _, final_result = cv2.split(blackAndWhiteImage)
final_result = final_result / 255
cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_BGR2RGB)
plt.imshow(blackAndWhiteImage,)