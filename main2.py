from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt

#  در این فایل مشکل فید کردن داده ها و اعمال فیلتر روی دیتا ست رو دارم
X_train, y_train = read_hoda_dataset(dataset_path='Train 60000.cdb',
                                     images_height=32,
                                     images_width=32,
                                     one_hot=False,
                                     reshape=True)

X_test, y_test = read_hoda_dataset(dataset_path='Test 20000.cdb',
                                   images_height=32,
                                   images_width=32,
                                   one_hot=False,
                                   reshape=True)

X_remaining, y_remaining = read_hoda_dataset(dataset_path='RemainingSamples.cdb',
                                             images_height=32,
                                             images_width=32,
                                             one_hot=False,
                                             reshape=True)



gray = cv2.cvtColor(X_train,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()