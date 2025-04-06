class AppConfig:
    CAMERA_ID = 0
    WIDTH = 1280
    HEIGHT = 720
    LEFT_EYE_LANDMARK_INDEXES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_LANDMARK_INDEXES = [42, 43, 44, 45, 46, 47]
    DLIB_LANDMARK_MODEL = "resources/shape_predictor_68_face_landmarks.dat"
    MODEL_FILE_NAME = "ml/models.joblib"
    LOAD_MODEL = False
    TRAINING_ITERATIONS = 20
    N_COMPONENTS = 3
    RANDOM_STATE = 42
    WARM_START = True
    OUTPUT_FILE_NAME = "files/eye_gaze_direction.txt"
