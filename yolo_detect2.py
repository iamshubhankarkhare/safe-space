import numpy as np
import math
import argparse
import time
import cv2
import os
from gtts import gTTS
from time import sleep
import os
import pyglet





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

path=r'/home/shubhankar/Desktop/yoloDetect/vid3.mp4'
vidObj = cv2.VideoCapture(path) 
frameRate = vidObj.get(5)



def speak(text):
	tts = gTTS(text=text, lang='en',slow=False)
	filename = '/tmp/temp.mp3'
	tts.save(filename)

	music = pyglet.media.load(filename, streaming=False)
	music.play()
	sleep(music.duration) 
	os.remove(filename) 



uniqueObj=set()
objToCall=set()
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
	
	# print("took {:.6f} secs".format(end - start))
	
	
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


	setToSpeak=uniqueObj-objToCall
	toSpeak="detected"
	if setToSpeak:
		speak("detected")
	for i in setToSpeak:
		if i != "car":
			toSpeak=toSpeak+" "+i
			speak(i)
	if(toSpeak != "detected"):
		print(toSpeak)
		# speak(toSpeak)

	objToCall.update(uniqueObj)
	uniqueObj.clear()




image = cv2.imread(args["image"])
cv2.waitKey(0)
# detect(image)
while vidObj.isOpened():
	success, frame = vidObj.read()
	frameId = vidObj.get(1)
	if success == False :
		break
	# cv2.imshow('Frame',frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


	if frameId % (math.floor(frameRate)*2) == 0:
		# cv2.imshow('Frame',frame)
		# print(objToCall)
		detect(frame)
	if frameId % (math.floor(frameRate)*24) == 0:
		objToCall.clear()
		print("clear")

vidObj.release()
cv2.waitKey(0)
cv2.destroyAllWindows()



