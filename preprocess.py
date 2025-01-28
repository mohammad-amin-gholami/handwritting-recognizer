import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('3.jpg',0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img,(32,32))

# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
#
# sharpened = cv2.filter2D(img, -1, kernel)
# cv2.imshow('gray', gray)

edges = cv2.Canny(img, 50, 150)

cv2.imshow("Edges", edges)
cv2.waitKey(0)

cv2.imshow('img',img)




# img_rgb = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
