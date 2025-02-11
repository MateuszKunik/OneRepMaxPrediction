import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset for handling sequence data with optional augmentation.
    """
    def __init__(self, data, max_frequency, augmentation=None):
        self.max_frequency = max_frequency
        self.augmentation = augmentation
        
        self.data = data
        self.num_coordinate_columns = count_coordinate_columns(self.data)
        self.num_coordinate_dimensions = count_coordinate_dimensions(self.data)
                                                                                
        # Prepare tensor storage
        self.tensor = torch.tensor([])
        
        tmp = self.data.groupby(by='FileId', as_index=False).size()
        tmp = tmp.rename(columns={'size': 'Frequency'})
        self.data = pd.merge(self.data, tmp)

        self.data['MaxFrequency'] = max_frequency

        for _, file_data in self.data.groupby(by='FileId'):
            # Drop the FileId column
            file_data = file_data.drop(columns='FileId')

            # Add padding to dataframe
            adjusted = self.add_padding(file_data)
            adjusted = adjusted.drop(columns=['Frequency', 'MaxFrequency'])
            
            # Convert the adjusted dataframe to a numpy array
            array = adjusted.to_numpy()

            # Convert numpy array to pytorch tensor 
            file_tensor = torch.from_numpy(array).unsqueeze(dim=0)
            file_tensor = file_tensor.to(torch.float32)
            # Concatenate to other tensors
            self.tensor = torch.cat((self.tensor, file_tensor), dim=0)


    def floor_ceil(self, x):
        return int(np.floor(x)), int(np.ceil(x))
    
    
    def add_padding(self, data):
        """
        Adds padding to the dataframe to match the maximum frequency.

        Args:
            data (pd.DataFrame): Dataframe containing the dataset.

        Returns:
            pd.DataFrame: Dataframe with added padding.
        """
        # Reset index
        data = data.reset_index(drop=True)

        # Calculate how much padding should be added
        difference = data.loc[0, 'MaxFrequency'] - data.loc[0, 'Frequency']

        if difference > 1:
            # Calculate how many padding should be added to the beginning and to the end
            front, back = self.floor_ceil(difference / 2)

            # Get the first and last record
            first_record, last_record = data.iloc[0], data.iloc[-1]

            # Prepare data frames
            to_beginning = pd.concat(front * [pd.DataFrame([first_record])])
            to_end = pd.concat(back * [pd.DataFrame([last_record])])

            # Return concatenated data frames
            return pd.concat([to_beginning, data, to_end], ignore_index=True)

        elif difference == 1:
            # Get only the last record
            last_record = data.iloc[-1]

            # Return concatenated data frames
            return pd.concat([data, pd.DataFrame([last_record])], ignore_index=True)

        else:
            return data


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.tensor.shape[0]
    

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Features and target of the sample.
        """
        # Get sample from data based on index
        sample = self.tensor[index, :, :]

        # Extract features and target
        features = sample[:, :-1]
        target = sample[0, -1].unsqueeze(0)

        dynamic_features = features[:, :self.num_coordinate_columns]
        dynamic_features = dynamic_features.reshape(
            self.num_coordinate_dimensions, self.max_frequency, -1, 1)
        
        static_features = features[0, self.num_coordinate_columns:]
                
        # Augment if necessary
        if self.augmentation:
            dynamic_features = self.augmentation(dynamic_features)
            

        return (dynamic_features, static_features), target
    

def count_coordinate_columns(dataframe):
    return extract_coordinate_data(dataframe).shape[1]


def count_coordinate_dimensions(dataframe):
    coordinate_columns = list(extract_coordinate_data(dataframe))
    dimensions = set(column[-1] for column in coordinate_columns)

    return len(dimensions)


def extract_coordinate_data(dataframe):
    return dataframe.filter(regex='X$|Y$|Z$')