import cv2
from matplotlib import pyplot as plt

video=cv2.VideoCapture("C:/empty/WIN_20131025_012927.MP4")
fps=video.get(cv2.cv.CV_CAP_PROP_FPS)
success,frame=video.read()
if success:
	plt.imshow(img)
	plt.show()