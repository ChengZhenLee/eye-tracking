from configs.app_config import AppConfig
import cv2
from file_handlers.file_handler import FileHandler
from handlers.face_handler import FaceHandler
from handlers.frame_handler import FrameHandler
from handlers.eye_behaviour_handler import EyeBlinkDetector, EyeGazeDetector
from ml.machine_learning import GMMHandler
from renderer.renderer import Renderer


class Trackers:
    def __init__(self, cap, detector, predictor, gmmModel, leftEyeLandmarkIndexes, rightEyeLandmarkIndexes, trainingIterations, outputFileName, loadModel, modelFileName):
        """
        Initialize with dependencies and configuration
        Args:
            config: AppConfig object containing:
                - training_iterations
                - left_eye_indexes
                - right_eye_indexes
        """
        self.cap = cap
        self.detector = detector
        self.predictor = predictor
        self.gmmModel = gmmModel
        self.leftEyeLandmarkIndexes = leftEyeLandmarkIndexes
        self.rightEyeLandmarkIndexes = rightEyeLandmarkIndexes
        self.trainingIterations = trainingIterations
        self.outputFileName = outputFileName
        self.modelFileName = modelFileName
        
        # Frame state
        self.frame = None
        self.gray = None
        self.current_face = None
        self.landmarks = None
        
        # Training state
        self.trained_iterations = 0
        self.is_training_complete = loadModel

    def trackFace(self):
        """Main tracking loop"""
        while not self._should_exit():
            self._process_frame()
            self._process_faces()
            self._update_display()

        cv2.destroyAllWindows()

    # Frame processing methods
    def _process_frame(self):
        """Capture and preprocess frame"""
        self.frame = FrameHandler.readCapture(self.cap)
        self.frame = FrameHandler.flipFrameHorizontally(self.frame)
        self.gray = FrameHandler.grayscaleFrame(self.frame)
    
    def _process_faces(self):
        """Handle all detected faces in frame"""
        faces = FaceHandler.detectFace(self.gray, self.detector)
        for face in faces:
            self._process_single_face(face)

    def _process_single_face(self, face):
        """Process individual face"""
        self.current_face = face
        self._extract_landmarks()
        self._track_eye_behavior()
        
        if not self.is_training_complete:
            self._handle_training()

    # Behavior tracking methods
    def _extract_landmarks(self):
        """Detect and store facial landmarks"""
        FaceHandler.drawFaceRegion(self.current_face, self.frame)
        self.landmarks = FaceHandler.predictFace(
            self.gray, 
            self.current_face, 
            self.predictor
        )

    def _track_eye_behavior(self):
        """Track both blinking and gaze"""
        if self.is_training_complete:
            self._track_eye_gaze()
        self._track_eye_blinking()

    def _track_eye_blinking(self):
        """Handle eye blink detection"""
        if EyeBlinkDetector.detectEyesBlinking(
            self.landmarks,
            self.leftEyeLandmarkIndexes,
            self.rightEyeLandmarkIndexes
        ):
            Renderer.renderEyeBlinking(self.frame)

    def _track_eye_gaze(self):
        """Handle eye gaze tracking"""
        gazeDir = EyeGazeDetector.predictGazeDirection(
            self.gmmModel,
            self.gray,
            self.landmarks,
            self.leftEyeLandmarkIndexes,
            self.rightEyeLandmarkIndexes
        )
        Renderer.renderEyeGazeDirection(self.frame, gazeDir)
        self._save_gaze_direction(gazeDir)

    # Training methods
    def _handle_training(self):
        """Manage GMM training process"""
        if self.trained_iterations < self.trainingIterations:
            self._collect_training_data()
        else:
            self._finalize_training()

    def _collect_training_data(self):
        """Store training samples"""
        Renderer.renderTraining(self.frame)
        GMMHandler.addDataPoint(
            self.gray,
            self.landmarks,
            self.leftEyeLandmarkIndexes,
            self.rightEyeLandmarkIndexes
        )
        self.trained_iterations += 1

    def _finalize_training(self):
        """Complete the training phase"""
        GMMHandler.fitDataPointsToModel(self.gmmModel)
        GMMHandler.saveModel(self.gmmModel, self.modelFileName)
        self.is_training_complete = True

    # Utility methods
    def _update_display(self):
        """Update output display"""
        cv2.imshow("Frame", self.frame)

    def _should_exit(self):
        """Check for exit condition"""
        return cv2.waitKey(30) == 27  # ESC key
    
    def _save_gaze_direction(self, gazeDir):#
        if gazeDir == 0:
            data = "right"
        elif gazeDir == 1:
            data = "center"
        else:
            data = "left"

        FileHandler.saveData(self.outputFileName, data)