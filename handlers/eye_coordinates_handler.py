import cv2
import dlib
from utils.math_helpers import CoordinatesMath, NpMath
import numpy as np


class EyeCoordinatesCalculator:
    def calculateEyeCoordinates(landmarks: dlib.full_object_detection, eyeLandmarkIndexes: list[int]) -> np.array:
        eyeCoords = np.array(
            [(landmarks.part(index).x, landmarks.part(index).y)
            for index in eyeLandmarkIndexes],
            dtype = np.int32
        )

        return eyeCoords
        
    def calculateEyeCrosshairCoords(landmarks: dlib.full_object_detection, eyeLandmarkIndexes: list[int]) -> np.ndarray:
        eyeCoords = EyeCoordinatesCalculator.calculateEyeCoordinates(landmarks, eyeLandmarkIndexes)

        eyeCrosshairBoundaries = np.array([
            eyeCoords[0],
            CoordinatesMath.midpoint(
                eyeCoords[1],
                eyeCoords[2]
            ),
            eyeCoords[3],
            CoordinatesMath.midpoint(
                eyeCoords[4],
                eyeCoords[5]
            )
        ])
        
        return eyeCrosshairBoundaries
    

class EyeRegionExtractor:
    def extractEyeRegion(frame: np.ndarray, landmarks: dlib.full_object_detection, eyeLandmarkIndexes: list[int]) -> np.ndarray:
        eyeCoords = EyeCoordinatesCalculator.calculateEyeCoordinates(landmarks, eyeLandmarkIndexes)

        # Create a mask with the same dimensions as the eye region
        # But the region outside the eye polygon is blacked out
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eyeCoords], True, 255, 1)
        cv2.fillPoly(mask, [eyeCoords], 255)

        eyeRegion = cv2.bitwise_and(frame, frame, mask=mask)

        # Get just the eye region in the frame
        corners = NpMath.getRegionCorner(eyeCoords)
        eyeRegion = eyeRegion[corners[2]:corners[3], corners[0]:corners[1]]

        return eyeRegion