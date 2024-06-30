# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:12:25 2024

@author: user
"""

import cv2


yüz_takip = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


image_path = r"C:\Users\user\Desktop\images\resim11.jpg"
image = cv2.imread(image_path)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


yüz = yüz_takip.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


for (x, y, w, h) in yüz:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)


cv2.imshow('Fotograf', image)


cv2.waitKey(0)
cv2.destroyAllWindows()