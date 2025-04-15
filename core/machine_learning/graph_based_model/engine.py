import torch
from tqdm.auto import tqdm

from .engine_setup import (
    initialize_model,
    initialize_training_components,
    initialize_callbacks
)

from .engine_utils import (
    get_device_from_model,
    transfer_data_to_device,
    log_training_start,
    log_epoch_results,
    initialize_results_tracker,
    log_training_params,
    log_training_metrics,
    update_results_tracker,
    is_stopper_triggered,
    log_early_stopping,
    log_training_complete,
    is_training_mode,
    get_context_manager,
    calculate_average,
    create_checkpoints
)


def setup_and_train_model(
        pose_landmarker,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        model_parameters: dict
):

    model = initialize_model(
        pose_landmarker, model_parameters["graph_parameters"], model_parameters)

    loss_fn, optimizer, lr_scheduler = initialize_training_components(
        model, model_parameters)

    init_stopper, early_stopper, model_checkpoint = initialize_callbacks(model_parameters)

    training_metrics = train_and_validate_model(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        init_stopper=init_stopper,
        early_stopper=early_stopper,
        num_epochs=model_parameters["num_epochs"])

    best_training_step = model_checkpoint.get_best_training_step()

    checkpoints = create_checkpoints(
        optimizer, lr_scheduler, training_metrics, best_training_step)

    return model, checkpoints


def train_and_validate_model(
        model, train_dataloader, valid_dataloader,
        loss_fn, optimizer, lr_scheduler, model_checkpoint,
        init_stopper=None, early_stopper=None, num_epochs=100,
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
        
        model_checkpoint.update_training_step(model, epoch, valid_loss)

    log_training_complete("regression_model", epoch+1)

    return results_tracker


def perform_training_step(model, dataloader, loss_fn, optimizer, lr_scheduler):
    return perform_step(model, dataloader, loss_fn, optimizer, lr_scheduler)


def perform_validation_step(model, dataloader, loss_fn):
    return perform_step(model, dataloader, loss_fn)


def perform_step(model, dataloader, loss_fn, optimizer=None, lr_scheduler=None):
    training_mode = is_training_mode(optimizer)
    device = get_device_from_model(model)
    accumulated_loss = 0.0

    model.train() if training_mode else model.eval()

    with get_context_manager(training_mode):
        for features, targets in dataloader:
            features, targets = transfer_data_to_device(features, targets, device)
            predictions = model(features)

            loss = loss_fn(predictions, targets)
            accumulated_loss += loss.item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()

                # \/ wrzucić do funkcji
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()

        if training_mode and lr_scheduler:
            lr_scheduler.step()

    average_loss = calculate_average(accumulated_loss, dataloader)

    return average_loss