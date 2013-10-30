from matplotlib import pyplot as plt
import cv2
webCamHndlr = cv2.VideoCapture(0)
ret,img = webCamHndlr.read()
plt.imshow(img)
plt.show()