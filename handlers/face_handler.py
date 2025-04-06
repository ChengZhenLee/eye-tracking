import cv2


class FaceHandler:
    def detectFace(frame, detector):
        faces = detector(frame)
        return faces
    
    def drawFaceRegion(face, frame):
        xLeft, yTop, xRight, yBottom = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (xLeft, yTop), (xRight, yBottom), (0, 255, 0), 3)

    def predictFace(frame, face, predictor):
        landmarks = predictor(frame, face)
        return landmarks