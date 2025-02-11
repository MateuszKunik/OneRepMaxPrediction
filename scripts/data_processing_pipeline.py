import os
import sys
sys.path.append("D:/DevSpace/Projects/DeepLearning/Research")

MAIN_PATH = "D:\DevSpace\Projects\DeepLearning\Research\data"
RAW_PATH = os.path.join(MAIN_PATH, "01_raw")
INTERMEDIATE_PATH = os.path.join(MAIN_PATH, "02_intermediate")
PRIMARY_PATH = os.path.join(MAIN_PATH, "03_primary")


from core.utils import (
    setup_configs,
    load_video_dataset,
    load_personal_data,
    handle_data_saving
)
from core.pose_estimation import (
    setup_custom_pose_landmarker
)
from core.data_processing import (
    process_video_to_dataframe,
    process_squat_and_personal_data
)


pose_params, data_params, model_params = setup_configs()

# pose landmarker tiny pipeline
custom_pose = setup_custom_pose_landmarker(pose_params)


# data processing pipeline
squat_video_files = load_video_dataset()
personal_data = load_personal_data()

squat_dataframe = process_video_to_dataframe(
    custom_pose_landmarker=custom_pose,
    squat_video_files=squat_video_files,
    parameters=data_params["extraction_parameters"]
)

handle_data_saving(
    dataframe=squat_dataframe,
    target_path=INTERMEDIATE_PATH,
    file_name="squat_data"
)

processed_data = process_squat_and_personal_data(
    squat_data=squat_dataframe,
    personal_data=personal_data,
    parameters=data_params["processing_parameters"]
)

handle_data_saving(
    dataframe=processed_data,
    target_path=PRIMARY_PATH,
    file_name="model_input_data",
    pose_params=pose_params,
    data_params=data_params
)