import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept) / slope)
    x2 = int((y2-intercept) / slope)

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = [] #  coordinates of the line which will display left
    right_fit = [] # // right
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0] # 기울기
        intercept = parameters[1] # 절편

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis = 0) # axis = 0은 x(row) , 1은 y(column) , 2는 z(depth)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,  (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        #for line in lines:
            #x1, y1, x2, y2 = line.reshape(4) # 2차원을 1차원으로 [[x1, y1, x2, y2]] -> [x1, y1, x2, y2]
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

image = cv2.imread('test_image.jpg')
if image is None:
    print('error')
    exit(1)

# vlane_image = np.copy(image)
# canny_image = canny(vlane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#
# averaged_lines = average_slope_intercept(vlane_image, lines)
#
# m_line_image = display_lines(vlane_image, averaged_lines)
# combo_image = cv2.addWeighted(vlane_image, 0.8, m_line_image, 1, 1) # 1번이미지, 가중치, 2번이미지, 2번이미지가중치,두 합에 또 더할 값
# cv2.imshow('result', combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    averaged_lines = average_slope_intercept(frame, lines)

    m_line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, m_line_image, 1, 1) # 1번이미지, 가중치, 2번이미지, 2번이미지가중치,두 합에 또 더할 값
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 1 millisecond in between frames . 0으로 하면 무한히 대기해야함
        break

cap.release()
cv2.destroyAllWindows()
