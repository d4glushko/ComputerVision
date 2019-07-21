import cv2
import numpy as np

size = (800,800,3)
white_background = np.ones(size,np.uint8) * 255
title = "kalman filter"

def mousemove(event, x, y, s, p):
    global white_background, current_measurement, current_prediction
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    current_prediction = kalman.predict()

    cm_x, cm_y = current_measurement[0], current_measurement[1]
    cp_x, cp_y = current_prediction[0], current_prediction[1]

    white_background = np.ones(size,np.uint8) * 255
    red = (0, 0, 255)
    green = (0, 255, 0)
    cv2.putText(white_background, "Current measurement: {:.1f}, {:.1f}".format(np.float(cm_x), np.float(cm_y)),
                (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, red)
    cv2.putText(white_background, "Prediction: {:.1f}, {:.1f}".format(np.float(cp_x), np.float(cp_y)),
                (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, green)
    cv2.circle(white_background, (cm_x, cm_y), 10, red, -1)
    cv2.circle(white_background, (cp_x, cp_y), 10, green, -1)

    kalman.correct(current_measurement)
    return


cv2.namedWindow(title)
cv2.setMouseCallback(title, mousemove)

kalman = cv2.KalmanFilter(4,2,0)

# measure only x and y, without v_x and v_y
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)

#  transition matrix =
#  [1 0 dT 0]
#  [0 1 0 dT]
#  [0 0 1  0]
#  [0 0 0  1]
kalman.transitionMatrix = np.array([[1, 0, 0.2, 0],[0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

#   [Ex 0  0    0]
#   [0 Ey  0    0]
#   [0 0 E_v_x  0]
#   [0 0  0  E_v_y]
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001

while True:
    cv2.imshow(title, white_background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()