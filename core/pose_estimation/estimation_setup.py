from .pose_landmarker import CustomPoseLandmarker

def setup_custom_pose_landmarker(parameters: dict):
    return CustomPoseLandmarker(
        selected_values=parameters["selected_values"],
        custom_landmarks=parameters["custom_landmarks"]
    )