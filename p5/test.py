import cv2 
import numpy as np
import pandas as pd

if __name__ == '__main__':
    img = cv2.imread('screenshot.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    startPos = [625,92]
    img = img[max(0,startPos[1]*2-300):min(300+startPos[1]*2,img.shape[1]),max(0,startPos[0]*2-300):min(300+startPos[0]*2,img.shape[0])]
    if(img.shape[0]<600 or img.shape[1]<600):
        new_img = np.zeros((600,600))
        start_x = (600 - img.shape[0]) // 2
        start_y = (600 - img.shape[1]) // 2
        new_img[start_x:start_x + img.shape[0],start_y:start_y + img.shape[1]] = img
        cv2.imwrite('screenshot.png',new_img)
        print(img.shape)
    else:
        cv2.imwrite('screenshot.png',img)
        print(img.shape)
    