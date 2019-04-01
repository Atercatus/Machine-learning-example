import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle # 텍스트 이외의 자료형을 파일로 저장학 위해 사용하는 모듈(Seriallizable, Deseriallizable)
import pandas as pd # python data analysis library
import random
import cv2

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# histogram equalizer
def equalize(img):
    img = cv2.equalizeHist(img) # only accept gray scale img
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

np.random.seed(0)

with open('C:/Users/AterCatus/Desktop/ML_Examples/Self_driving Car/Classifying Road Symbols/german-traffic-signs/train.p', 'rb') as f: #rb: reading binary
    train_data = pickle.load(f)
with open('C:/Users/AterCatus/Desktop/ML_Examples/Self_driving Car/Classifying Road Symbols/german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('C:/Users/AterCatus/Desktop/ML_Examples/Self_driving Car/Classifying Road Symbols/german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

X_train, Y_train = train_data['features'], train_data['labels']
X_val, Y_val = val_data['features'], val_data['labels']
X_test, Y_test = test_data['features'], test_data['labels']

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

assert(X_train.shape[0] == Y_train.shape[0]), "The number of images is not equal to the number of labels"
assert(X_val.shape[0] == Y_val.shape[0]), "The number of images is not equal to the number of labels"
assert(X_test.shape[0] == Y_test.shape[0]), "The number of images is not equal to the number of labels"
assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32x 3"
assert(X_val.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32x 3"
assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32x 3"

data = pd.read_csv('C:/Users/AterCatus/Desktop/ML_Examples/Self_driving Car/Classifying Road Symbols/german-traffic-signs/signnames.csv')
print(data)


num_of_samples = []

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols= cols, figsize=(5, 100))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows(): #(index, Series)
        print(j)
        print(row)
        x_selected = X_train[Y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis('off')
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["SignName"])
            num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
# plt.show()

plt.imshow(X_train[1000])
plt.axis("off")
# plt.show()

img = equalize(grayscale(X_train[1000]))
plt.imshow(img)
plt.show()

#map은 iterator 를 반환한다 따라서 iterator -> lis 화 해주고 이를 array로 변경한다
X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)

Y_train = to_categorical(Y_train, 43)
Y_test = to_categorical(Y_test, 43)
Y_val = to_categorical(Y_val, 43)

def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    # Complie model
    model.compile(Adam(lr=0.01), loss="categorical_crossentropy", metrics=['accuracy'])
    return model

def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation="relu"))
    model.add(Conv2D(60, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # prevent overfitting
    # conv층 뒤에 위치하는 것이 좋음
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    # prevent overfitting
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    # Complie model
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
    return model

# model = leNet_model()
# print(model.summary())
model = modified_model()
print(model.summary())
# 이전 모델과 비교했을 때 전체적인 파라미터 수가 줄었다
history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val), batch_size=400, verbose = 1, shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, Y_test, verbose = 0)

print('Test Score:', score[0])
print('Test Accuracy:', score[1])

#fetch image

import requests
from PIL import Image
url = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))


#Preprocess image
img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap = plt.get_cmap('gray'))
print(img.shape)

#Reshape reshape
img = img.reshape(1, 32, 32, 1)

#Test image
print("predicted sign: "+ str(model.predict_classes(img)))
