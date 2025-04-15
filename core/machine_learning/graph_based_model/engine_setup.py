from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from .model_builder import STGCNModel
from .loss import RMSELoss, ConstrainedLoss
from .callbacks import InitStopper, EarlyStopper, ModelCheckpoint



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
    
    if model_parameters["lr_scheduler"] == "step":
        lr_scheduler = StepLR(
            optimizer,
            step_size=model_parameters["step_size"],
            gamma=model_parameters["gamma"])
        
    elif model_parameters["lr_scheduler"] == "cosine":
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
    
    model_checkpoint = ModelCheckpoint(
        maximize=model_parameters["maximize"])
    
    return init_stopper, early_stopper, model_checkpoint