import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

points = 500
X = np.linspace(-3, 3, points)
Y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=1))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1))

adam = Adam(lr=0.01)
# mse: mean square error
model.compile(loss="mse", optimizer=adam)
model.fit(X, Y, epochs=50)

prediction = model.predict(X)
plt.scatter(X, Y)
plt.plot(X, prediction, 'ro')
plt.show()
