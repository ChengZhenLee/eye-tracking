from configs.app_config import AppConfig
import cv2
from core.trackers import Trackers
import dlib
from ml.machine_learning import GMMHandler

class TrackersFactory:
    def createTrackers(appConfig: AppConfig) -> Trackers:
        cap = cv2.VideoCapture(appConfig.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, appConfig.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, appConfig.HEIGHT)
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(appConfig.DLIB_LANDMARK_MODEL)

        model = None
        if appConfig.LOAD_MODEL:
            model = GMMHandler.loadModel(appConfig.MODEL_FILE_NAME)
        if model == None:
            model = GMMHandler.initialiseModel(n_components=appConfig.N_COMPONENTS, random_state=appConfig.RANDOM_STATE, warm_start=appConfig.WARM_START)
            
        trackers = Trackers(cap, detector, predictor, model, appConfig.LEFT_EYE_LANDMARK_INDEXES, appConfig.RIGHT_EYE_LANDMARK_INDEXES, appConfig.TRAINING_ITERATIONS, appConfig.OUTPUT_FILE_NAME, appConfig.LOAD_MODEL, appConfig.MODEL_FILE_NAME)

        return trackers