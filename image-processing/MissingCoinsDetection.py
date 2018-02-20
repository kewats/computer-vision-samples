import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('tray-of-coins.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

box_mask = cv2.inRange(hsv, np.array([0,90,190]), np.array([180,255,255]))

box_mask = cv2.bitwise_not(box_mask)
box_mask = cv2.medianBlur(box_mask, 15)
cv2.imshow('mask', box_mask)

coin_mask = cv2.inRange(hsv, np.array([5, 0, 0]), np.array([170, 255, 255]))
coin_mask = cv2.medianBlur(coin_mask, 25)
cv2.imshow('mask2', coin_mask)

_, contours, _ = cv2.findContours(coin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# print str(len(contours)) + ' boxes found'

for con in contours:
    M = cv2.moments(con)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    h, w = box_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(box_mask, mask, (cX,cY), 0)

cv2.imshow('filled', box_mask)

_, final_contours, _ = cv2.findContours(box_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
missing = 0
for con in final_contours:
    x,y,w,h = cv2.boundingRect(con)
    if x != 0 | y != 0:
        missing += 1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# print str(missing) + ' coins missing'
cv2.putText(img,'Missing:' + str(missing),(10,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
