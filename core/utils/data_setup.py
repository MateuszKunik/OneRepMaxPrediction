import os
import pandas as pd

from .dataset_manager import VideoDatasetManager
from .utilities import (
    generate_directory_path,
    ensure_directory_exists,
    save_dataframe_to_csv,
    save_config_params,
    log_dataframe_saved,
    log_params_saved,
    log_saved_file_path
)


MAIN_PATH = "D:\DevSpace\Projects\DeepLearning\Research\data"

RAW_PATH = os.path.join(MAIN_PATH, "01_raw")
INTERMEDIATE_PATH = os.path.join(MAIN_PATH, "02_intermediate")
PRIMARY_PATH = os.path.join(MAIN_PATH, "03_primary")


def load_video_dataset() -> list:
    dataset = VideoDatasetManager(
        folder_path=os.path.join(RAW_PATH, "squat_videos"))

    return dataset.get_video_files()


def load_personal_data() -> pd.DataFrame:
    return pd.read_excel(
        io=os.path.join(RAW_PATH, "PersonalData.xlsx"))


def handle_data_saving(
        dataframe: pd.DataFrame,
        target_path: str,
        file_name: str,
        pose_params: dict = None,
        data_params: dict = None,
) -> None:
    """
    opis
    """
    directory_path = generate_directory_path(target_path, file_name)
    ensure_directory_exists(directory_path)

    save_dataframe_to_csv(directory_path, dataframe, file_name)
    log_dataframe_saved(directory_path, file_name)

    if pose_params:
        config_file_name = "pose_parameters.txt"
        save_config_params(directory_path, pose_params, config_file_name)
        log_params_saved(directory_path, config_file_name)

    if data_params:
        config_file_name = "data_parameters.txt"
        save_config_params(directory_path, data_params, config_file_name)
        log_params_saved(directory_path, config_file_name)
    
    log_saved_file_path(directory_path)


def load_model_input_data(data_version: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(PRIMARY_PATH, f"{data_version}/model_input_data.csv")
    )