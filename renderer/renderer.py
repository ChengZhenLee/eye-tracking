import cv2
import numpy as np

class Renderer:
    def renderEyeCrosshair(eyeCrosshairBoundaries: np.ndarray, frame: np.ndarray, colour = (0, 0, 255), thickness = 1):
        cv2.line(frame, eyeCrosshairBoundaries[0], eyeCrosshairBoundaries[2], colour, thickness)
        cv2.line(frame, eyeCrosshairBoundaries[1], eyeCrosshairBoundaries[3], colour, thickness)

    def renderEyeBlinking(frame):
        cv2.putText(frame, "BLINKING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    def renderEyeGazeDirection(frame: np.ndarray, gazeDirection: int):
        # 0: right, blue
        # 1: center, green
        # 2: left, red
        if gazeDirection == 0:
            frame[:] = (255, 0, 0)
            cv2.putText(frame, "RIGHT", (600, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif gazeDirection == 1:
            frame[:] = (0, 255, 0)
            cv2.putText(frame, "CENTER", (600, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif gazeDirection == 2:
            frame[:] = (0, 0, 255)
            cv2.putText(frame, "LEFT", (600, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def renderTraining(frame):
        cv2.putText(frame, "LEFT", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "CENTER", (600, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT", (1180, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
