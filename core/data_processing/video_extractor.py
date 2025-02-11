import os
import cv2
import random
import numpy as np
import pandas as pd
from itertools import product
from typing import Union, List
from mediapipe import solutions

from .custom_pose_landmarks import CustomPoseLandmark
from .utils import get_custom_landmarks, landmark2array
from .utils import VIDEO_FEATURE_KEYS


                                                                                
class VideoLandmarksExtractor():
    def __init__(
            self,
            custom_pose: CustomPoseLandmark,
            feature_keys: Union[List[str], bool] = True
    ) -> pd.DataFrame:
        """
        opis klasy
        """
        self.mp_pose = solutions.pose
        self.custom_pose = custom_pose
        self.feature_keys = resolve_feature_keys(feature_keys)


    def extract_video_landmarks(
            self,
            file_path: str,
            model_complexity: int,
            min_detection_confidence: float,
            min_tracking_confidence: float
    ) -> list:
        """
        opis
        """
        features = extract_video_feature_values(file_path, self.feature_keys)
        log_conversion_start(features['FileId'])

        frame_records = []
        cap = cv2.VideoCapture(file_path)
        pose_estimator = self.initialize_pose_estimator(
            model_complexity, min_detection_confidence, min_tracking_confidence)

        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break
            
            processed_image = self.prepare_image_for_estimation(image)
            estimation_results = pose_estimator.process(processed_image)
            frame_record = self.generate_frame_record(
                features, estimation_results)
            
            frame_records.append(frame_record)
                
        pose_estimator.close()
        cap.release()
        cv2.destroyAllWindows()

        return frame_records

    def generate_landmarks_dataframe(
            self,
            source: list,
            num_samples: int = None,
            model_complexity: int = 1,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ) -> pd.DataFrame:
        """
        opis metody
        """
        all_landmarks_records = []

        source = random.sample(source, num_samples) if num_samples else source

        for file_path in source:
            video_records = self.extract_video_landmarks(
                file_path=file_path,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            all_landmarks_records.extend(video_records)
            
        
        dataframe = pd.DataFrame(
            all_landmarks_records, columns=self.prepare_column_names())

        return dataframe
    

    def prepare_column_names(self) -> list:
        return self.feature_keys + self.generate_landmark_columns()
    

    def generate_landmark_columns(self) -> list:
        landmarks = self.custom_pose.get_dictionary().values()

        return [
            ''.join([landmark.title().replace('_', ''), axis])
            for landmark, axis in product(landmarks, ['X', 'Y', 'Z'])]
    

    def initialize_pose_estimator(
            self,
            model_complexity: int,
            detection: float,
            tracking: float
    ) -> solutions.pose.Pose:
        return self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=detection,
            min_tracking_confidence=tracking)
    

    def prepare_image_for_estimation(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (360, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        return image


    def generate_frame_record(self, features: dict, results) -> list:
        record = list(features.values())
        
        if results.pose_landmarks:
            record += self.extract_landmark_coordinates(results.pose_landmarks)
        else:
            record += self.fill_missing_landmarks(record)
            log_file_warning(features['FileId'])
        
        return record
    

    def extract_landmark_coordinates(self, pose_landmarks) -> list:
        custom_pose_landmarks = get_custom_landmarks(
            self.custom_pose, pose_landmarks)

        coordinates = []
        for landmark in custom_pose_landmarks.landmark:
            coordinates.extend(landmark2array(landmark)[:3])

        return coordinates


    def fill_missing_landmarks(self, record: list) -> list:
        #return np.zeros(len(columns) - len(record)).tolist()
        return np.zeros(57).tolist()







# FUNKCJE POZA METODA VIDEOLANDMARKSEXTRACTOR
import logging
logger = logging.getLogger("data_processing_logger")

def log_file_warning(file_id: str) -> None:
    logger.warning(f"Please check {file_id} file.")
    print(f"Please check {file_id} file.")


def log_conversion_start(file_id: str) -> None:
    logger.info(f"Extracting pose landmarks from {file_id} file...")
    print(f"Extracting pose landmarks from {file_id} file...")


def resolve_feature_keys(feature_keys: Union[list, bool]) -> list:
    if isinstance(feature_keys, list):
        return feature_keys

    elif isinstance(feature_keys, bool):
        return VIDEO_FEATURE_KEYS if feature_keys else [VIDEO_FEATURE_KEYS[0]]


def extract_video_feature_values(
        file_path: str, required_features: list) -> dict:
    """
    opis
    dodać przykład file_name
    wyjasnic konstrukcje file_name
    """
    file_name = extract_file_name(file_path)
    feature_values = parse_feature_values(file_name)
    converted_values = convert_feature_values(feature_values)

    # the fourth index always corresponds to the load
    converted_values[4] = adjust_load_value(converted_values[4])
    
    features_dict = create_features_dict(file_name, converted_values)

    return filter_features(features_dict, required_features)


def extract_file_name(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def parse_feature_values(file_name: str) -> list:
    return file_name.split('_')


def convert_feature_values(feature_values: list) -> list:
    numeric_features = convert_numeric_feature_values(feature_values[:-1])
    camera_position = convert_camera_position_value(feature_values[-1])

    return numeric_features + [camera_position]


def convert_numeric_feature_values(feature_values: list) -> list:
    return list(map(int, feature_values))


def convert_camera_position_value(camera_position: str) -> str:
    position_mapping = {'L': 'left', 'C': 'center', 'R': 'right'}

    return position_mapping[camera_position]


def adjust_load_value(load_value: int) -> float:
    """
    opis
    wyjasnic dlaczego dodajemy 0.5kg
    """
    if load_value % 5 != 0:
        load_value += 0.5
    
    return load_value


def create_features_dict(file_name: str, values: list) -> dict:
    return dict(zip(VIDEO_FEATURE_KEYS, [file_name] + values))


def filter_features(features_dict: dict, required_keys: list) -> dict:
    return dict(
        filter(lambda item: item[0] in required_keys, features_dict.items())
    )