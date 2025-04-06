from handlers.eye_behaviour_handler import EyeGazeDetector
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture


class GMMHandler:
    dataPoints = []

    @staticmethod
    def initialiseModel(n_components, random_state, warm_start) -> GaussianMixture:
        gmmModel = GaussianMixture(n_components=n_components, random_state=random_state, warm_start=warm_start)

        return gmmModel
    
    @staticmethod
    def fitDataPointsToModel(gmmModel):
        data = np.array(GMMHandler.dataPoints).reshape(-1, 1)
        gmmModel.fit(data)

        order = np.argsort(gmmModel.means_.ravel())
        gmmModel.means_ = gmmModel.means_[order]
        gmmModel.weights_ = gmmModel.weights_[order]
        gmmModel.covariances_ = gmmModel.covariances_[order]
        gmmModel.precisions_ = gmmModel.precisions_[order]
        gmmModel.precisions_cholesky_ = gmmModel.precisions_cholesky_[order]

    @staticmethod
    def addDataPoint(frame: np.ndarray, landmarks, leftEyeLandmarkIndexes: list[int], rightEyeLandmarkIndexes: list[int]):
        GMMHandler.dataPoints.append(EyeGazeDetector.getAverageGazeRatio(frame, landmarks, leftEyeLandmarkIndexes, rightEyeLandmarkIndexes))

    @staticmethod
    def saveModel(gmmModel, fileName):
        joblib.dump(gmmModel, fileName)
    
    @staticmethod
    def loadModel(fileName):
        try:
            gmmModel = joblib.load(fileName)
            return gmmModel
        except FileNotFoundError:
            print(f"No saved model found at {fileName}")
            return None
        
        

