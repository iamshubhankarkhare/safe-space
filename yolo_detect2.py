import numpy as np
import math
import argparse
import time
import cv2
import os



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

path=r'/home/shubhankar/Desktop/yoloDetect/vid2.mp4'
vidObj = cv2.VideoCapture(path) 
frameRate = vidObj.get(5)


uniqueObj=set()
def detect(image):
	(H, W) = image.shape[:2]

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	
	print("YOLO took {:.6f} seconds".format(end - start))
	
	
	boxes = []
	confidences = []
	classIDs = []
	
	
	for output in layerOutputs:
	
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
	
			if confidence > args["confidence"]:
				confidences.append(float(confidence))
				classIDs.append(classID)

	for i in classIDs:
		uniqueObj.add(LABELS[i])

	print(uniqueObj)
	uniqueObj.clear()




image = cv2.imread(args["image"])
cv2.waitKey(0)
# detect(image)
while vidObj.isOpened():
	success, frame = vidObj.read()
	frameId = vidObj.get(1)
	if success == False :
		break

	if frameId % (math.floor(frameRate)*2) == 0:
		# cv2.imshow('Frame',frame)
		print(frameId)
		detect(frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

vidObj.release()
cv2.destroyAllWindows()



