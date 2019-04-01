import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.models import Model
import random

np.random.seed(0)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

assert(X_train.shape[0] == Y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."
assert(X_test.shape[0] == Y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."

num_of_samples=[]

cols = 5
num_classes = 10

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_classes):
      x_selected = X_train[Y_train == j]
      axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
      axs[j][i].axis("off")
      if i == 2:
        axs[j][i].set_title(str(j))
        num_of_samples.append(len(x_selected))

# print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

X_train = X_train/255
X_test = X_test/255

# define the leNet_model function
def leNet_model():
    model = Sequential()
    # 30개의 filter, 각 필터는 5 by 5, input은 28 by 28의 depth가 1인 이미지 => 30개의 feature map 생성
    # padding은 사용하지 않는다 => 왜냐하면 우리 샘플 이미지의 가장자리에는 관심이 없기 때문
    # strides 는 default value(1)
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    # Dropout layer는 parameter의 수가 많은 곳에 위차한다
    # => overfitting의 가능성이 높기 때문
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(Adam(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = leNet_model()
history = model.fit(X_train, Y_train, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)

plt.figure(figsize=(16, 4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=10, batch_size=200, verbose=1, shuffle=1)
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('Loss')
plt.xlabel('epoch')

plt.subplot(1,2,2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# GET image on web and cal
import requests
from PIL import Image# python imaging library
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcST8KzXHtkSHcxzdpnllMhAj0upLEwnNFdtY6j4YUPcmaf4Ty3u'
response = requests.get(url, stream=True)
print(response)
img = Image.open(response.raw)

import cv2

img_array = np.asarray(img)
#(850, 850, 4) => (28, 28, 4) // 4: r,g,b,alpha
resized = cv2.resize(img_array, (28, 28))
#(28, 28, 4) => (28, 28)
gray_scale= cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap=plt.get_cmap("gray"))
plt.show()

# nomalization
image = image/255
image = image.reshape(1, 28, 28, 1)
# for multiclass
prediction = model.predict_classes(image)
print("Predicted digit:", str(prediction))

layer1 = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
layer2 = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)
visual_layer1, visual_layer2 = layer1.predict(image), layer2.predict(image)
print(visual_layer1.shape)
print(visual_layer2.shape)

# plt for first cnn layer
plt.figure(figsize=(10, 6))
for i in range(30):
    plt.subplot(6, 5, i+1)
    plt.imshow(visual_layer1[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')
plt.show()

# plt for second cnn layer
plt.figure(figsize=(10, 6))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(visual_layer2[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')
plt.show()
