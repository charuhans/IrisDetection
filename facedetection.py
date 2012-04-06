#!/usr/bin/python

#load system libraries
from SimpleCV import *
from operator import add
from SimpleCV import Color, ColorCurve, Image, pg, np, cv, ImageClass
from SimpleCV.Display import Display
import sys

#extact haar cascades from the xml file
def getHaarFeatures(fileName = None):
	if fileName is None:
		print "Please provide the file name"
		return
	haarCascade = cv.Load(fileName)
	return haarCascade

#this method used cv.HoughCicles fo detection od iris, it did not wok fo any of the input images	
def detectCircles(im = None):
	if im is None:
		print "Please give a valid parameter"
		return
		
	im = cv.LoadImage('1.jpg')
	gray = cv.CreateImage(cv.GetSize(im), 8, 1)
	edges = cv.CreateImage(cv.GetSize(im), 8, 1)

	cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
	cv.Canny(gray, edges, 20, 55, 3)

	storage = cv.CreateMat(im.width, 1, cv.CV_32FC3)
	cv.HoughCircles(edges, storage, cv.CV_HOUGH_GRADIENT, 5, 25, 200, 10)

	for i in xrange(storage.width - 1):
		radius = storage[i, 2]
		center = (storage[i, 0], storage[i, 1])
		print (radius, center)
		cv.Circle(im, center, radius, (0, 0, 255), 3, 8, 0)

	cv.NamedWindow('Circles')
	cv.ShowImage('Circles', im)
	cv.WaitKey(0)

#detect face using haar featues
def detectFace(img = None, face_cascade = None):
	if face_cascade is None:
		print 'Face cascade is not sent in as parameter'
		return
	
	if img is None:
		print 'Face image is not sent in as parameter'
		return
		
	faces = img.findHaarFeatures(face_cascade)
		
	faces = faces.sortArea();
	face = faces[-1]
	myFace = face.crop()
	
	return myFace, face.x, face.y, face.width, face.height

#detect eye region using haar featues
def detectEyes(img = None, eye_cascade = None):
	if eye_cascade is None:
		print 'Eye cascade is not sent in as parameter'
		return
	
	if img is None:
		print 'mage is not sent in as parameter'
		return
		
	eyes = img.findHaarFeatures(eye_cascade)
	eyes  = eyes.sortArea()
	eye1 = eyes[-1];
	eye2 = eyes[-2];
	
	return eye1, eye1.x, eye1.y, eye2, eye2.x, eye2.y

#detect iris region by detecting blobs
def captureIris(EYE = None, fileName = None):
	EYE1 = EYE.crop()
	greyEYE1 = EYE1.grayscale()                    			#convert to grayscale
	greyEYE1.smooth("gaussian",(3,3), 2, grayscale = True)	#do gaussian smoothing
	
	#blob way of gettin iris
	blob1 = BlobMaker()

	blobs = blob1.extractFromBinary(greyEYE1.invert().binarize(180).invert(),greyEYE1)

	if(len(blobs)>0):
		#blobs[0].drawHull(color=(0, 255, 0), alpha=-1, width=-1, layer=None)
		blobs[0].drawOutline(color=(255, 0, 0), alpha=-1, width=1, layer=None)
		radius = blobs[0].radius()

	if fileName is None:
		greyEYE1.save("eye_image.jpg")
		return
		
	greyEYE1.save(fileName)
	
	return radius
	
#input image of your choice here
img = Image("input1.jpg")	

#get the cascade for eyes
eye_cascade  = getHaarFeatures("haarcascade_eye.xml")

#get the cascade for frontal face
face_cascade = getHaarFeatures("haarcascade_frontalface_default.xml")

#face image and its starting x and y co-ordinate relative to original image
(face, face_x, face_y, face_w, face_h) = detectFace(img, face_cascade)

#get the infomation about the two eyes that ae detected
(EYE1, eye1_x, eye1_y, EYE2, eye2_x, eye2_y) = detectEyes(face, eye_cascade) 

#detect blobs within the eye region to capture iris, capture radius of blob for eye1
eye1_radius = captureIris(EYE1, 'eye1.jpg')

#detect blobs within the eye region to capture iris, capture radius of blob for eye 2
eye2_radius = captureIris(EYE2, 'eye2.jpg')

#draw cicle on the cropped face image to illustrate the detected iris region
face.drawCircle((eye1_x, eye1_y), eye1_radius, color=(0, 255, 0), thickness=3)

#draw cicle on the cropped face image to illustrate the detected iris region
face.drawCircle((eye2_x, eye2_y), eye1_radius, color=(255, 0, 0), thickness=3)

#save image with circles drawn as a result
face.save("MyImage.jpg")