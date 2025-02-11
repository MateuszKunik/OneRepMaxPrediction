import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model_builder import STGCNModel
from .loss import RMSELoss, ConstrainedLoss
from .callbacks import InitStopper, EarlyStopper

from .engine_utils import (
    get_device_from_model,
    transfer_to_device,
    log_training_start,
    log_epoch_results,
    initialize_results_tracker,
    log_training_params,
    log_training_metrics,
    update_results_tracker,
    is_stopper_triggered,
    log_early_stopping,
    log_training_complete,
)


def setup_and_train_model(
        pose_landmarker,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        model_parameters: dict
) -> torch.nn.Module:
    
    graph_parameters = model_parameters["graph_parameters"]

    regression_model = initialize_model(
        pose_landmarker, graph_parameters, model_parameters)

    loss_fn, optimizer, lr_scheduler = initialize_training_components(
        regression_model, model_parameters)

    init_stopper, early_stopper = initialize_callbacks(model_parameters)

    results = train_and_validate_model(
        model=regression_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        init_stopper=init_stopper,
        early_stopper=early_stopper,
        num_epochs=model_parameters["num_epochs"])

    return regression_model, optimizer, lr_scheduler, results
    

def initialize_model(custom_pose, graph_parameters, model_parameters):
    graph_parameters = {
        "skeleton_layout": custom_pose, **graph_parameters}

    model = STGCNModel(
        in_channels=model_parameters["channels"],
        graph_args=graph_parameters,
        edge_importance_weighting=model_parameters["edge_importance"],
        dropout=model_parameters["dropout"])

    return model.to(model_parameters["device"])


def initialize_training_components(model, model_parameters):
    loss_fn = initialize_loss_fn(model_parameters)
    
    optimizer = AdamW(
        model.parameters(),
        lr=model_parameters["learning_rate"],
        weight_decay=model_parameters["weight_decay"])
    
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=model_parameters["t_max"],
        eta_min=model_parameters["eta_min"])    
    
    return loss_fn, optimizer, lr_scheduler


def initialize_loss_fn(model_parameters):
    loss_fn = ConstrainedLoss(
        base_loss=RMSELoss(),
        penalty_weight=model_parameters["penalty_weight"])
    
    return loss_fn
    

def initialize_callbacks(model_parameters):
    init_stopper = InitStopper(
        patience=model_parameters["init_stopper_patience"])
    
    early_stopper = EarlyStopper(
        patience=model_parameters["early_stopper_patience"],
        min_delta=model_parameters["early_stopper_min_delta"])
    
    return init_stopper, early_stopper


def train_and_validate_model(
        model, train_dataloader, valid_dataloader, loss_fn,
        optimizer, lr_scheduler, init_stopper=None, early_stopper=None,
        num_epochs=100,
):

    results_tracker = initialize_results_tracker()

    log_training_start()
    log_training_params(num_epochs, optimizer, lr_scheduler)

    for epoch in tqdm(range(num_epochs)):
        train_loss = perform_training_step(
            model, train_dataloader, loss_fn, optimizer, lr_scheduler)

        valid_loss = perform_validation_step(model, valid_dataloader, loss_fn)

        log_epoch_results(epoch, train_loss, valid_loss)
        log_training_metrics(train_loss, valid_loss, epoch)
        update_results_tracker(results_tracker, train_loss, valid_loss)

        if is_stopper_triggered(init_stopper, valid_loss):
            log_early_stopping("init_stopper")
            break

        if is_stopper_triggered(early_stopper, valid_loss):
            log_early_stopping("early_stopper")
            break

    log_training_complete("regression_model", epoch+1)

    return results_tracker


def perform_training_step(model, dataloader, loss_fn, optimizer, lr_scheduler):
    model.train()
    device = get_device_from_model(model)

    accumulated_loss = 0.0

    for features, targets in dataloader:
        features, targets = transfer_to_device(features, targets, device)
        predictions = model(features)

        loss = loss_fn(targets, predictions)
        accumulated_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

    lr_scheduler.step()

    accumulated_loss = accumulated_loss / len(dataloader)

    return accumulated_loss


def perform_validation_step(model, dataloader, loss_fn):
    model.eval()
    device = get_device_from_model(model)

    accumulated_loss = 0.0

    with torch.inference_mode():
        for features, targets in dataloader:
            features, targets = transfer_to_device(features, targets, device)
            predictions = model(features)

            loss = loss_fn(targets, predictions)
            accumulated_loss += loss.item()

    accumulated_loss = accumulated_loss / len(dataloader)

    return accumulated_loss


def evaluate_model_performance(model, dataloader, model_parameters):
    model.eval()
    device = get_device_from_model(model)
    loss_fn = initialize_loss_fn(model_parameters)
    accumulated_loss = 0.0

    with torch.inference_mode():
        for features, targets in dataloader:
            features, targets = transfer_to_device(features, targets, device)
            predictions = model(features)

            loss = loss_fn(targets, predictions)
            accumulated_loss += loss.item()

    accumulated_loss = accumulated_loss / len(dataloader)

    return accumulated_loss