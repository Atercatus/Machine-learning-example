import numpy as np
import keras
from keras.models import Sequential # linear stack of layers
from keras.layers import Dense # fully connected NN
from keras.optimizers import Adam
#Dense will be used to construct densely-connected NN layers
import matplotlib.pyplot as plt
#%matplotlib inline
# magic function
#이는 IPython 에서 제공하는 Rich output 에 대한 표현 방식인데요,
#도표와 같은 그림, 소리, 애니메이션 과 같은 결과물들을 Rich output 이라 합니다

n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts), # 평균 13, 분산 2, 갯수
               np.random.normal(12, 2, n_pts)]).T # 벡터화
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T

X = np.vstack((Xa, Xb)) # vertical stack
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

#plt.scatter(X[:n_pts,0], X[:n_pts,1]) # ([0부터 n_pts 까지, 0], [n_pts부터 0까지]) // X ,Y
#plt.scatter(X[n_pts:,0], X[n_pts:,1])

model = Sequential() # linear stack of layers
model.add(Dense(units = 1, input_shape=(2,), activation='sigmoid')) #units: #hidden layers
adam = Adam(lr = 0.1)
model.compile(adam, loss='binary_crossentropy', metrics = ['accuracy'])
#binary_crossentropy: 내가 아는 그 logistric regression의 Loss fn
# metrics : 측정 방법
#학습 시작
h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs = 500, shuffle='true') # verbose: 학습 중 출력된느 문구를 설정
#verbose: 로깅 (0: 없음, 1: 프로그레스바, 2: epoch당)
#shuffle: 각 epoch에 따라 training data set을 shuffle 할 것인가.
plt.subplot(2,2,1)
plt.plot(h.history['acc'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'])

plt.subplot(2,2,2)
plt.plot(h.history['loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.legend(['losss'])
#plt.show()

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

    pred_func = model.predict(grid)
    print("pred_func", pred_func)
    z = pred_func.reshape(xx.shape)
    print("z", z)
    plt.subplot(2,2,3)
    plt.contourf(xx,yy,z)

# just plot decision boundary


#test
#plt.subplot(2,2,3)
plt.scatter(X[:n_pts,0], X[:n_pts,1]) # ([0부터 n_pts 까지, 0], [n_pts부터 0까지]) // X ,Y
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 7.5
y = 5
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker="o", markersize=10, color="red")
plt.show()
print("Prediction is :", prediction)
