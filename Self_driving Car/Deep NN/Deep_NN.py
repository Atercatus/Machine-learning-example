import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

n_pts = 500
X, Y = datasets.make_circles(n_samples=n_pts, random_state = 123, noise=0.1, factor=0.2)
# noise: Standard deviation of Gaussian noise added to the data
# random_state: Determines random number generation for dataset shuffling and noise
# factor: Scale factor between inner and outer circle
#plt.figure(figsize=(15,15))
plt.rcParams["figure.figsize"] = (10, 5)

plt.subplot(2,2,1)
plt.scatter(X[Y==0, 0], X[Y==0, 1])
plt.scatter(X[Y==1, 0], X[Y==1, 1])

model = Sequential()
model.add(Dense(units = 5, input_shape=(2,), activation='sigmoid'))
# model.add(Dense(units = 1, activation='sigmoid'))
# input 에서 이미 input_shape를 지정해 줬으므로 이후에 input_shape를 지정해 줄 필요가 없다.
model.add(Dense(units = 1, input_shape=(5,), activation='sigmoid'))
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=Y, verbose=1, batch_size=20, epochs=100, shuffle='true')

def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25, 50)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

plt.subplot(2,2,3)
plot_decision_boundary(X, Y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])


plt.subplot(2,2,2)
plt.plot(h.history['acc'], label='accuracy')
plt.plot(h.history['loss'], label='loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='best')

plt.subplot(2,2,3)
x = 0.1
y = 0
point = np.array([[x, 0]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='red')
print("Prediction is : ", prediction)
plt.show()
