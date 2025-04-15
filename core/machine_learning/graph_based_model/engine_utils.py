import os
import torch
import torch.nn as nn
import mlflow
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .engine_setup import initialize_loss_fn

from core.utils.utilities import (
    generate_directory_path,
    ensure_directory_exists,
    save_config_params,
    log_params_saved,
    log_saved_file_path)


def get_device_from_model(model):
    return next(model.parameters()).device


def transfer_to_device(tensor, device):
    return tensor.to(device)
    

def transfer_data_to_device(features, targets, device):
    dynamic_features, static_features = features
    dynamic_features = transfer_to_device(dynamic_features, device)
    static_features = transfer_to_device(static_features, device)

    targets = transfer_to_device(targets, device)

    return (dynamic_features, static_features), targets


def log_training_start():
    print("Training has started...\n")


def log_epoch_results(epoch, train_loss, valid_loss):
    print(f"\nEpoch: {epoch}")
    print(f"Training loss: {train_loss:.4f} | Validation loss: {valid_loss:.4f}\n")


def plot_loss_curves(losses: dict) -> plt.Figure:
    num_epochs = len(losses["train_loss"])
    epoch_range = range(1, num_epochs + 1)
    
    fig, ax = plt.subplots()
    ax.plot(epoch_range, losses["train_loss"], label="Training Loss Curve")
    ax.plot(epoch_range, losses["valid_loss"], label="Validation Loss Curve")
    ax.set_title("Loss Curves for Training and Validation Data")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Root Mean Square Error (RMSE)")
    ax.grid(True)
    ax.legend()

    return fig


def prepare_directory(target_path: str, directory_name: str) -> str:
    directory_path = generate_directory_path(target_path, directory_name)
    ensure_directory_exists(directory_path) 

    return directory_path


def handle_training_artifacts_saving(
        model: nn.Module,
        checkpoints: dict,
        training_parameters: dict,
        evaluation_report: dict,
        target_path: str,
) -> None:
    """
    opis
    """
    directory_path = prepare_directory(
        target_path, training_parameters["model_name"])
    
    log_model_artifacts(directory_path)

    save_model(directory_path, model)
    log_model_saved()

    save_checkpoints(directory_path, checkpoints)
    log_checkpoints_saved()

    training_metrics = checkpoints["training_metrics"]
    loss_figure = plot_loss_curves(training_metrics)
    save_plot(directory_path, loss_figure, file_name="loss.png")

    print_training_summary(training_metrics)
    save_training_summary(directory_path, training_metrics)

    save_evaluation_report(directory_path, evaluation_report)

    config_file_name = "model_parameters.txt"
    save_config_params(directory_path, training_parameters, config_file_name)
    log_params_saved(config_file_name)

    log_saved_file_path(directory_path)


def save_model(target_dir, model, file_name="model.pth"):
    torch.save(model.state_dict(), os.path.join(target_dir, file_name))


def log_model_saved():
    print("Model has been successfully saved.")

    
def create_checkpoints(optimizer, scheduler, metrics, best_step) -> dict:
    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "training_metrics": metrics,
        "best_training_step": best_step
    }


def save_checkpoints(target_dir, checkpoints, file_name="my_checkpoints.pth"):
    torch.save(checkpoints, os.path.join(target_dir, file_name))


def log_checkpoints_saved():
    print("Checkpoints have been successfully saved.")


def save_plot(target_dir, figure, file_name="plot.png"):
    figure.savefig(os.path.join(target_dir, file_name))


def log_plot_saved():
    print("Plot has been successfully saved.")


def initialize_results_tracker() -> dict:
    return {
        "train_loss": [],
        "valid_loss": [],
    }


def setup_mlflow(parameters: dict) -> None:
    mlflow.set_tracking_uri(parameters["tracking_uri"])
    mlflow.set_experiment(parameters["experiment_name"])


def log_training_params(num_epochs, optimizer, lr_scheduler) -> None:
    mlflow.log_params({
        "num_epochs": num_epochs,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "scheduler": str(lr_scheduler),
    })


def log_training_metrics(
    train_loss: float, valid_loss: float, num_epochs: int) -> None:
    mlflow.log_metrics(
        metrics={
            "train_loss": train_loss,
            "valid_loss": valid_loss
        },
        step=num_epochs
    )


def log_model_artifacts(target_path: str) -> None:
    mlflow.log_artifacts(target_path)


def update_results_tracker(
    results: dict, train_loss: float, valid_loss: float) -> None:
    results["train_loss"].append(train_loss)
    results["valid_loss"].append(valid_loss)


def is_stopper_triggered(stopper, valid_loss: float) -> bool:
    return stopper and stopper.stop(valid_loss)


def log_early_stopping(stopper_name: str) -> None:
    print(f"Training stopped early due to '{stopper_name}' condition.")


def log_training_complete(model_name: str, total_epochs: int) -> None:
    print(f"Training of '{model_name}' completed after {total_epochs} epochs.\n")


def print_training_summary(results: dict) -> None:
    print("-- -- Training Summary: -- --")
    print(f"Number of Epochs: {len(results['train_loss'])}\n")

    print(f"Final Training Loss: {results['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {results['valid_loss'][-1]:.4f}\n")


def is_training_mode(optimizer) -> bool:
    return bool(optimizer)


def get_context_manager(training_mode):
    return torch.set_grad_enabled(True) if training_mode else torch.inference_mode()


def calculate_average(metric, dataloader):
        return metric / len(dataloader)


def generate_evaluation_report(model, dataloader, best_training_step, parameters):
    evaluation_report = evaluate_model_performance(model, dataloader, "final")
    print_evaluation_report(evaluation_report)
    
    # zrobić funkcje z pobierania najleppszej epoki \/
    best_epoch = best_training_step["epoch"]

    if best_epoch < parameters["num_epochs"]:
        # zrobić funkcję z komunikatu \/
        print(f"The best model weights from epoch {best_epoch} have been loaded.\n")
        model.load_state_dict(best_training_step["weights"])

        evaluation_report = evaluate_model_performance(model, dataloader, "best")
        print_evaluation_report(evaluation_report)
    
    return evaluation_report


def evaluate_model_performance(model, dataloader, evaluation_stage):
    loss_fn = initialize_loss_fn()
    targets, predictions = fetch_targets_and_predictions(model, dataloader)

    report = prepare_classification_report(targets, predictions, loss_fn)
    report["evaluation_stage"] = evaluation_stage

    return report


def fetch_targets_and_predictions(model, dataloader):
    device = get_device_from_model(model)
    model.eval()

    all_targets, all_predictions = [], []

    with torch.inference_mode():
        for features, targets in dataloader:
            features, targets = transfer_data_to_device(features, targets, device)
            predictions = model(features)

            all_targets.append(targets)
            all_predictions.append(predictions)

    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    return all_targets, all_predictions


def prepare_classification_report(targets, predictions, loss_fn):
    loss = loss_fn(predictions, targets).item()
    targets = transfer_to_device(targets, "cpu")
    predictions = transfer_to_device(predictions, "cpu")

    return {
        "loss": loss,
        "MAE": mean_absolute_error(targets, predictions),
        "MSE": mean_squared_error(targets, predictions),
        "r2_score": r2_score(targets, predictions)
    }


def print_evaluation_report(report, loss_function_name):
    print("-- -- Model Evaluation -- --")
    print(f"-- Evaluation Stage: {report['evaluation_stage']} --")
    print(f"Loss ({loss_function_name}): {report['loss']:.4f}\n")

    print(f"Mean Absolute Error: {report['MAE']:.4f}")
    print(f"Mean Squared Error: {report['MSE']:.4f}")
    print(f"R2 Score: {report['r2_score']:.4f}\n")


def save_training_summary(target_dir, training_metrics):
    with open(os.path.join(target_dir, "training_metrics.pkl"), "wb") as file:
        pickle.dump(training_metrics, file)


def save_evaluation_report(target_dir, evaluation_report):
    with open(os.path.join(target_dir, "evaluation_report.pkl"), "wb") as file:
        pickle.dump(evaluation_report, file)