import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

np.random.seed(0)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

assert(X_train.shape[0] == Y_train.shape[0]), "The number of image is not equal to the number of labels."
assert(X_test.shape[0] == Y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28 by 28."
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28 by 28."

num_of_samples = []

cols = 5
num_classes = 10 # 0 ~ 9
#
# fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
# fig.tight_layout() # overlap 해결
# for i in range(cols):
#     for j in range(num_classes):
#         x_selected = X_train[Y_train == j]
#         axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), : , :], cmap=plt.get_cmap("gray"))
#         axs[j][i].axis("off")
#         if i == 2:
#             axs[j][i].set_title(str(j))
#             num_of_samples.append(len(x_selected))
#
# plt.show()
# plt.figure(figsize=(12, 4))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("Distribution of the training dataset")
# plt.xlabel("Class number")
# plt.ylabel("Number of images")
# plt.show()

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
# nomalization
X_train = X_train/255 # 0 ~ 1 val
X_test = X_test/255

# 28 X 28 의 2-dimension 을 784의 1-dimansion 으로 바꿔야한다
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(Adam(lr=0.01), loss="categorical_crossentropy", metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())

plt.figure(figsize=(16, 4))
plt.subplot(1,2,1)
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=30, batch_size=200, verbose=1, shuffle=1)
plt.plot(history.history['loss'])
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

score = model.evaluate(X_test, Y_test, verbose=0)
print("Test score", score[0])
print("Test accuracy", score[1])

# GET image on web and cal
import requests
from PIL import Image# python imaging library
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
print(response)
img = Image.open(response.raw)
# plt.imshow(img)
# plt.show()
import cv2

img_array = np.asarray(img)
#(850, 850, 4) => (28, 28, 4) // 4: r,g,b,alpha
resized = cv2.resize(img_array, (28, 28))
#(28, 28, 4) => (28, 28)
gray_scale= cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)
# plt.imshow(image, cmap=plt.get_cmap("gray"))
# plt.show()

# nomalization
image = image/255
image = image.reshape(1, 784)
# for multiclass
prediction = model.predict_classes(image)
print("Predicted digit:", str(prediction))
# 정확도가 높진 않음 => CNN을 사용하는 이유!
