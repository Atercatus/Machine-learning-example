import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    ln = plt.plot(x1, x2, '-')
    plt.pause(0.0001)
    ln[0].remove()


def sigmoid(score):
    return 1/(1 + np.exp(-score)) # 확률 (probability를 구하는 공식)

def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(points * line_parameters) # probability를 구하는 공식
    cross_entropy = -(1/m)*(np.log(p).T * y + np.log(1-p).T*(1-y))
    return cross_entropy

def gradient_descent(line_parameters, points, y, alpha): #alpha is traning rate
    m = points.shape[0]
    for i in range(500):
        p = sigmoid(points*line_parameters)
        gradient = (points.T* (p-y))*(alpha/m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0) # np.array 에서 첫번째 아이템 반환
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + (x1*(-w1/w2))
        # w1x1 + w2x2 + b = 0 // decision boundary
        # draw(x1, x2)
        # print(calculate_error(line_parameters, points, y))


n_pts = 150
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

#sigmoid code
w1 = -0.2
w2 =  -0.35
b = 3.5
line_parameters = np.matrix([w1, w2, b]).T
#end

#line_parameters = np.matrix([np.zeros(3)]).T
#print(line_parameters)

#w1x1 + w2x2 + b = 0
#x2 = -b / w2 + w1 * (-w1 / w2)

#sigmoid code
# x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
# x2 = -b / w2 + (x1*(-w1/w2))
linear_combination = all_points*line_parameters
#print(linear_combination)
probabilities = sigmoid(linear_combination)
print(probabilities)
#end

#cross enctropy
# y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2,1)
# 위에 존재하는 점은 0을 아래에 존재하는 점은 1을 // y의 값

# fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(top_region[:, 0], top_region[:,1], color='r')
# ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
# gradient_descent(line_parameters, all_points, y, 0.06)
# draw(x1, x2)
# plt.show()
