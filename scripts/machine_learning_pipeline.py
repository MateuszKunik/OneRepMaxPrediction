import sys

if sys.platform == "win32":
    sys.path.append("D:/DevSpace/Projects/Research/OneRepMaxPrediction")
elif sys.platform == "linux":
    sys.path.append("/mnt/d/gniazdko/OneRepMaxPrediction")

import mlflow
from core.utils import ProjectManager
from core.utils import setup_configs, load_model_input_data
from core.pose_estimation import setup_custom_pose_landmarker

from core.machine_learning.graph_based_model import (
    setup_mlflow,
    filter_dataframe,
    split_data_by_proportions,
    create_dataloaders,
    setup_and_train_model,
    generate_evaluation_report,
    handle_training_artifacts_saving,
)

project_manager = ProjectManager()
primary_data_path = project_manager.get_primary_data_path()
model_directory_path = project_manager.get_model_data_path()

pose_parameters, data_parameters, model_parameters = setup_configs()

pose_landmarker = setup_custom_pose_landmarker(pose_parameters)
model_input_data = load_model_input_data(model_parameters["data_version"])

setup_mlflow(model_parameters["mlflow_parameters"])

with mlflow.start_run():
    filtered_model_input_data = filter_dataframe(
        model_input_data, model_parameters["data_parameters"])

    data_splits = split_data_by_proportions(
        filtered_model_input_data, model_parameters["data_parameters"])

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        filtered_model_input_data,
        data_splits,
        model_parameters["data_parameters"])

    regression_model, checkpoints = setup_and_train_model(
        pose_landmarker,
        train_dataloader,
        valid_dataloader,
        model_parameters["model_parameters"])
    
    evaluation_report = generate_evaluation_report(
        regression_model,
        test_dataloader,
        checkpoints["best_training_step"],
        model_parameters["model_parameters"])
    
    handle_training_artifacts_saving(
        regression_model,
        checkpoints,
        model_parameters,
        evaluation_report,
        model_directory_path)