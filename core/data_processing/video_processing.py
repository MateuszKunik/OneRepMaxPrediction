import pandas as pd

from .video_extractor import VideoLandmarksExtractor


def process_video_to_dataframe(
        custom_pose_landmarker,
        squat_video_files: list,
        parameters: dict
) -> pd.DataFrame:
    """
    opis
    """

    extractor = VideoLandmarksExtractor(
        custom_pose=custom_pose_landmarker,
        feature_keys=parameters["feature_keys"]
    )

    dataframe = extractor.generate_landmarks_dataframe(
        source=squat_video_files,
        num_samples=parameters["num_samples"],
        model_complexity=parameters["model_complexity"],
        min_detection_confidence=parameters["min_detection_confidence"],
        min_tracking_confidence=parameters["min_tracking_confidence"]
    )

    return dataframe
