import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_dataframe(dataframe, parameters):
    target_threshold = parameters["target_threshold"]
    filter_ratio = parameters["filter_ratio"]

    dataframe = filter_dataframe_by_target(
        dataframe, target_threshold)

    dataframe = filter_dataframe_by_ratio(
        dataframe, filter_ratio)

    return dataframe


def filter_dataframe_by_target(dataframe, target_threshold):
    if target_threshold != 0:
        dataframe = dataframe[
            dataframe["PercentageMaxLoad"] >= target_threshold]

    return dataframe


def filter_dataframe_by_ratio(dataframe, filter_ratio):
    if filter_ratio != 1:
        unique_data = get_unique_file_ids(dataframe)
        num_samples = int(len(unique_data) * filter_ratio / 2)
        
        first_samples = get_first_samples_of_dataframe(dataframe, num_samples)
        last_samples = get_last_samples_of_dataframe(dataframe, num_samples)

        dataframe = pd.concat([first_samples, last_samples]).drop_duplicates()

    return dataframe


def get_unique_file_ids(dataframe):
    return dataframe['FileId'].drop_duplicates()


def get_first_samples_of_dataframe(dataframe, num_samples):
    sorted_dataframe = sort_dataframe(dataframe)
    unique_dataframe = get_unique_file_ids(sorted_dataframe)
    first_unique_file_ids = unique_dataframe.head(num_samples)

    first_samples = sorted_dataframe.merge(
        first_unique_file_ids, how='inner', on='FileId')
    
    return first_samples


def get_last_samples_of_dataframe(dataframe, num_samples):
    sorted_dataframe = sort_dataframe(dataframe)
    unique_dataframe = get_unique_file_ids(sorted_dataframe)
    last_unique_file_ids = unique_dataframe.tail(num_samples)

    last_samples = sorted_dataframe.merge(
        last_unique_file_ids, how='inner', on='FileId')
    
    return last_samples


def sort_dataframe(dataframe):
    return dataframe.sort_values(by='PercentageMaxLoad', ascending=True)


def split_data_by_proportions(dataframe, parameters):
    proportions = np.array(parameters["split_proportions"])

    if not validate_proportions(proportions):
        raise ValueError("Proportions must add up to 1.")
    
    unique_file_ids = get_unique_file_ids(dataframe).values
    train_ids, validation_ids, test_ids = split_file_ids(
        unique_file_ids, proportions, parameters["seed"])
    
    return {"train": train_ids, "validation": validation_ids, "test": test_ids}


def validate_proportions(proportions):
    return np.round(proportions.sum(), 2) == 1


def split_file_ids(unique_file_ids, proportions, seed):
    valid_test_share = proportions[1:].sum()
    test_share = proportions[2] / valid_test_share

    train_ids, remaining_ids = train_test_split(
        unique_file_ids, test_size=valid_test_share, random_state=seed)

    validation_ids, test_ids = train_test_split(
        remaining_ids, test_size=test_share, random_state=seed)
    
    return train_ids, validation_ids, test_ids
