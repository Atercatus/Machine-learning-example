import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

def plot_decision_boundary(X, Y, model):
    x_span = np.linspace(min(X[:, 0])-1, max(X[:, 0])+1, 50)
    y_span = np.linspace(min(X[:, 1])-1, max(X[:, 1])+1, 50)
    xx, yy = np.meshgrid(x_span, y_span) # xx 는 [123][123][123]
                                         # yy 는 [111][222][333]
    xx_, yy_ = xx.ravel(), yy.ravel() # 1-D array로 만든다 order 존재
    grid = np.c_[xx_, yy_] #add along second axis. 1차원이라서 second axis가 없으면 만든다
    # ex) np.array([1.2.3]) => np.array([[1],[2],[3]])
    #print(np.c_[np.array([1,2,3]), np.array([4,5,6])])
    #print(np.c_[np.array([[1],[2],[3]]), np.array([[4],[5],[6]])])
    # 둘다 같은 결과 나옴

    # pred_func = model.predict(grid) -> 바이너리
    pred_func = model.predict_classes(grid) # -> 멀티
    # print("pred_func", pred_func)
    z = pred_func.reshape(xx.shape)
    # print("z", z)
    plt.contourf(xx,yy,z)
    # z is height

n_pts = 500
centers = [[-1, 1], [-1, -1], [1, -1]]
# make_blobs 함수는 등방성 가우시안 정규분포를 이용해 가상 데이터를 생성한다.
# 이 때 등방성이라는 말은 모든 방향으로 같은 성질을 가진다는 뜻이다
X, Y = datasets.make_blobs(n_samples = n_pts, random_state = 123, centers=centers, cluster_std=0.4)
# n_samples : 표본 데이터의 수, 디폴트 100
# n_features : 독립 변수의 수, 디폴트 20
# centers : 생성할 클러스터의 수 혹은 중심, [n_centers, n_features] 크기의 배열. 디폴트 3
# cluster_std: 클러스터의 표준 편차, 디폴트 1.0
# center_box: 생성할 클러스터의 바운딩 박스(bounding box), 디폴트 (-10.0, 10.0))
# print(X) # x,y value
# print(Y) # 3 classes

# one hot encoding
y_cat = to_categorical(Y, 3)

model = Sequential()
model.add(Dense(units = 3, input_shape=(2,), activation='softmax'))
model.compile(Adam(0.1), loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X, y=y_cat, verbose=1, batch_size = 50, epochs = 100)

plot_decision_boundary(X, y_cat, model)
plt.scatter(X[Y==0, 0], X[Y==0, 1])
plt.scatter(X[Y==1, 0], X[Y==1, 1])
plt.scatter(X[Y==2, 0], X[Y==2, 1])
# print([Y==0]) : [array([True, False,....])] 이런식으로 출력됨
# print(X[([False, False, True])]) : 3번쨰 값만 출력된다

x = 0.5
y = -1
point = np.array([[x,y]])
prediction = model.predict_classes(point)
plt.plot([x], [y], marker='o', markersize=10, color='r')
print("Prediction is", prediction)

plt.show()
