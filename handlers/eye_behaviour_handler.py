import cv2
import dlib
from handlers.eye_coordinates_handler import EyeCoordinatesCalculator, EyeRegionExtractor
import numpy as np
from sklearn.mixture import GaussianMixture
import sys
from utils.math_helpers import CoordinatesMath

class EyeBlinkDetector():
    def detectEyesBlinking(landmarks, leftEyeLandmarkIndexes, rightEyeLandmarkIndexes) -> bool:
        return EyeBlinkDetector.detectEyeBlinking(landmarks, leftEyeLandmarkIndexes) and EyeBlinkDetector.detectEyeBlinking(landmarks, rightEyeLandmarkIndexes)
    
    def detectEyeBlinking(landmarks: dlib.full_object_detection, eyeLandmarkIndexes) -> bool:
        eyeCrosshairCoords = EyeCoordinatesCalculator.calculateEyeCrosshairCoords(landmarks, eyeLandmarkIndexes)

        horizontalDist = CoordinatesMath.distance(eyeCrosshairCoords[0], eyeCrosshairCoords[2])
        verticalDist = CoordinatesMath.distance(eyeCrosshairCoords[1], eyeCrosshairCoords[3])
        distRatio = verticalDist / horizontalDist
        
        return distRatio < 0.25

class EyeGazeDetector():
    def getGazeRatio(frame: np.ndarray, landmarks, eyeLandmarkIndexes) -> float:
        eyeRegion = EyeRegionExtractor.extractEyeRegion(frame, landmarks, eyeLandmarkIndexes)
        eyeRegion = cv2.resize(eyeRegion, None, fx=5, fy=5)
        threshold = cv2.adaptiveThreshold(eyeRegion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        height, width = threshold.shape
        
        thresholdLeftSide = threshold[0:height, 0:width//2]
        thresholdRightSide = threshold[0:height, width//2:width]

        # count non-white pixels on the left and right side of the eye
        leftCount = cv2.countNonZero(thresholdLeftSide)
        rightCount = cv2.countNonZero(thresholdRightSide)
        if rightCount == 0:
            if leftCount == 0:
                # entire eye is black
                return 1
            # entire right-side of the eye is black
            return sys.maxsize
        
        return leftCount / rightCount

    def getAverageGazeRatio(frame: np.ndarray, landmarks, leftEyeLandmarkIndexes, rightEyeLandmarkIndexes) -> float:
        leftGazeRatio = EyeGazeDetector.getGazeRatio(frame, landmarks, leftEyeLandmarkIndexes)
        rightGazeRatio = EyeGazeDetector.getGazeRatio(frame, landmarks, rightEyeLandmarkIndexes)

        return (leftGazeRatio + rightGazeRatio) / 2

    def predictGazeDirection(gmmModel: GaussianMixture, frame, landmarks, leftEyeLandmarkIndexes, rightEyeLandmarkIndexes) -> int:
        averageGazeRatio = EyeGazeDetector.getAverageGazeRatio(frame, landmarks, leftEyeLandmarkIndexes, rightEyeLandmarkIndexes)
        dataPoint = np.array([[averageGazeRatio]])

        prediction = gmmModel.predict(dataPoint)

        # 0: left, 1: center, 2: right
        return prediction[0]