import mlflow
import sys
sys.path.append("D:/DevSpace/Projects/DeepLearning/Research")

from core.utils import ProjectManager
from core.utils import (
    setup_configs,
    load_video_dataset,
    load_personal_data,
    handle_data_saving,
)
from core.pose_estimation import (
    setup_custom_pose_landmarker
)
from core.data_processing import (
    process_video_to_dataframe,
    process_squat_and_personal_data
)
from core.machine_learning.graph_based_model import (
    setup_mlflow,
    filter_dataframe,
    split_data_by_proportions,
    create_dataloaders,
    setup_and_train_model,
    plot_loss_curves,
    summarize_training,
    handle_model_saving,
    evaluate_model_performance,
    prepare_directory,
    log_model_artifacts
)

project = ProjectManager()
model_directory_path = project.get_model_data_path()
intermediate_data_path = project.get_intermediate_data_path()
primary_data_path = project.get_primary_data_path()

pose_params, data_params, model_params = setup_configs()

custom_pose = setup_custom_pose_landmarker(pose_params)


# data processing pipeline
squat_video_files = load_video_dataset()
personal_data = load_personal_data()

squat_dataframe = process_video_to_dataframe(
    custom_pose, squat_video_files, data_params["extraction_parameters"])

handle_data_saving(
    squat_dataframe, intermediate_data_path, "squat_data")

model_input_data = process_squat_and_personal_data(
    squat_dataframe, personal_data, data_params["processing_parameters"])

handle_data_saving(
    model_input_data,
    primary_data_path,
    "model_input_data",
    pose_params, data_params
)

# machine learning pipeline
setup_mlflow(model_params["mlflow_parameters"])

with mlflow.start_run():
    filtered_model_input_data = filter_dataframe(
        model_input_data, model_params["data_parameters"])

    file_ids_by_split = split_data_by_proportions(
        filtered_model_input_data, model_params["data_parameters"])

    train_data, valid_data, test_data = create_dataloaders(
        filtered_model_input_data,
        file_ids_by_split,
        model_params["data_parameters"]
    )

    model, optimizer, lr_scheduler, results = setup_and_train_model(
        custom_pose, train_data, valid_data, model_params["model_parameters"])

    evaluation = evaluate_model_performance(
        model, test_data, model_params["model_parameters"])

    directory_path = prepare_directory(
        model_directory_path, model_params["model_name"])

    log_model_artifacts(directory_path)
    summarize_training(results, evaluation)

    figure = plot_loss_curves(results)

    handle_model_saving(
        model, optimizer, lr_scheduler, figure, directory_path, model_params)
