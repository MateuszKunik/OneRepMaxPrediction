import os
import yaml
import pandas as pd
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def generate_directory_path(target_path: str, prefix: str = None) -> str:
    timestamp = get_timestamp()
    directory_name = f"{prefix}_{timestamp}"
    return os.path.join(target_path, directory_name)


def ensure_directory_exists(target_path: str) -> str:
    os.makedirs(target_path, exist_ok=True)


def save_dataframe_to_csv(
        target_path: str,
        dataframe: pd.DataFrame,
        file_name: str
) -> None:
    dataframe.to_csv(os.path.join(target_path, f"{file_name}.csv"), index=False)


def log_dataframe_saved(file_name: str) -> None:
    print(f"DataFrame '{file_name}.csv' has been successfully saved.")


def save_config_params(target_path: str, parameters: dict, file_name: str) -> None:
    with open(os.path.join(target_path, file_name), "w") as file:
        yaml.dump(parameters, file, default_flow_style=False)


def log_params_saved(file_name: str) -> None:
    print(f"Configuration file '{file_name}' has been successfully saved.")


def log_saved_file_path(target_path: str) -> None:
    print(f"Path to saved file(s): {target_path}")