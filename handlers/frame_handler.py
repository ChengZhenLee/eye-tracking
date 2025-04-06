import cv2

class FrameHandler:
    def readCapture(cap):
        _, frame = cap.read()

        return frame
    
    def flipFrameHorizontally(frame):
        flipped = cv2.flip(frame, 1)

        return flipped
    
    def grayscaleFrame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return gray