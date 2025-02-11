from torch.utils.data import DataLoader
from .custom_dataset import CustomDataset
from .custom_transforms import GaussianNoise


def create_dataloaders(data, file_ids, parameters):
    max_sequence_length = calculate_max_sequence_length(data)

    train_dataloader = create_single_dataloader(
        data, file_ids, max_sequence_length,
        subset_type="train", parameters=parameters)
    
    valid_dataloader = create_single_dataloader(
        data, file_ids, max_sequence_length,
        subset_type="validation", parameters=parameters)
    
    test_dataloader = create_single_dataloader(
        data, file_ids, max_sequence_length,
        subset_type="test", parameters=parameters)
    
    return train_dataloader, valid_dataloader, test_dataloader


def calculate_max_sequence_length(data):
    return data.groupby(by='FileId').size().max()


def create_single_dataloader(
        data, file_ids, max_sequence_length, subset_type, parameters):
    data_subset = get_data_subset(data, file_ids, subset_type)

    if is_train_type(subset_type):
        augmentation = GaussianNoise(
            parameters["noise_probability"],
            parameters["noise_mean"],
            parameters["noise_std"])
                                                                                
        dataset = CustomDataset(
            data_subset, max_sequence_length, augmentation)
        shuffle = True
    else:
        dataset = CustomDataset(data_subset, max_sequence_length)
        shuffle=False
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=parameters["batch_size"],
        num_workers=parameters["num_workers"],
        pin_memory=parameters["pin_memory"],
        shuffle=shuffle)
    
    return dataloader


def get_data_subset(data, file_ids, subset_type):
    return data.loc[data['FileId'].isin(file_ids[subset_type])]


def is_train_type(subset_type):
    return subset_type == "train"