import pandas as pd
from tqdm import tqdm
from statsmodels.api import nonparametric as smnp
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def process_squat_and_personal_data(
        squat_data: pd.DataFrame,
        personal_data: pd.DataFrame,
        parameters: dict) -> pd.DataFrame :
    """
    opis
    """
    personal_data = calculate_BMI(personal_data)
    data = pd.merge(squat_data, personal_data, on='Id')

    data = prepare_regression_target(data)
    data = drop_irrelevant_columns(data)
    data = transform_features(data)
    data = smooth_coordinate_data(
        data, parameters["data_used_fraction"], parameters["num_iterations"])
    data = reorder_dataframe_columns(data)

    return data


def calculate_BMI(dataframe):
    data_copy = dataframe.copy()

    data_copy['Height_m'] = data_copy['Height'] / 100
    data_copy['BMI'] = data_copy['Weight'] / (data_copy['Height_m'] ** 2)

    del data_copy['Height_m']

    return data_copy


def extract_lifted_data(dataframe):
    return dataframe.loc[dataframe['Lifted'] == 1]


def calculate_max_load(dataframe):
    max_load_dataframe = dataframe[
        ['Id', 'Load']].groupby(by='Id', as_index=False).max()
    max_load_dataframe = max_load_dataframe.rename(columns={'Load': 'MaxLoad'})

    return pd.merge(dataframe, max_load_dataframe, on='Id')


def calculate_percentage_max_load(dataframe):
    dataframe['PercentageMaxLoad'] = 100 * dataframe['Load'] / dataframe['MaxLoad']
    
    return dataframe


def prepare_regression_target(dataframe):
    dataframe = extract_lifted_data(dataframe)
    dataframe = calculate_max_load(dataframe)
    dataframe = calculate_percentage_max_load(dataframe)

    return dataframe


def drop_irrelevant_columns(dataframe):
    irrelevant_columns = [
        'Id', 'Age', 'Height', 'Weight',
        'PastInjuries', 'LastInjury', 'PainDuringTraining',
        'SquatRecord', 'BenchPressRecord', 'DeadliftRecord',
        'PhysicalActivities', 'SetNumber', 'Load', 'Lifted', 'MaxLoad']
    
    return dataframe.drop(columns=irrelevant_columns)


def extract_coordinate_data(dataframe):
    return dataframe.filter(regex='X$|Y$|Z$')

def encode_categorical_features(dataframe, features):
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(dataframe[features])
    encoded_feature_names = encoder.get_feature_names_out(features)

    return pd.DataFrame(
        encoded_features,
        columns=encoded_feature_names,
        index=dataframe.index
    )


def normalize_continuous_features(dataframe, features):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_features = scaler.fit_transform(dataframe[features])
    
    return pd.DataFrame(
        normalized_features,
        columns=features,
        index=dataframe.index
    )

                                                                                
def transform_features(dataframe):
    categorical_features = [
        'ProficiencyLevel',
        'EquipmentAvailability',
        'TrainingProgram',
        'CameraPosition'
    ]
    continuous_features = [
        'TrainingFrequency',
        'TrainingExperience',
        'RepNumber',
        'Repetitions',
        'BMI'
    ]
    coordinate_features = list(extract_coordinate_data(dataframe))
    
    encoded_features = encode_categorical_features(
        dataframe, categorical_features)
    normalized_features = normalize_continuous_features(
        dataframe, continuous_features + coordinate_features)
    
    transformed_data = pd.concat(
        [
            dataframe.drop(columns=categorical_features + continuous_features + coordinate_features),
            encoded_features, normalized_features
        ], axis=1)
    
    return transformed_data


def smooth_with_lowess(column, frac, it):
    return smnp.lowess(
        column.values, column.index,
        frac=frac, it=it, return_sorted=False)


def smooth_coordinate_data(dataframe, frac, it):
    smoothed_dataframe_storage = []

    file_groups = dataframe.groupby(by='FileId')
    file_groups_tqdm = tqdm(
        file_groups, desc="Smoothing data", total=dataframe['FileId'].nunique())
    
    for _, file_data in file_groups_tqdm:
        coordinate_data = extract_coordinate_data(file_data)
        smoothed_data = coordinate_data.apply(smooth_with_lowess, args=(frac, it), axis=0)
        
        file_data = file_data.assign(**smoothed_data)
        smoothed_dataframe_storage.append(file_data)

    return pd.concat(smoothed_dataframe_storage, ignore_index=True)


def move_coordinates_to_start(dataframe):
    coordinate_columns = list(extract_coordinate_data(dataframe).columns)

    reordered_dataframe = dataframe[
        coordinate_columns + [col for col in dataframe if col not in coordinate_columns]]

    return reordered_dataframe


def move_percentage_to_end(dataframe):
    percentage_max_loads = dataframe.pop('PercentageMaxLoad')
    dataframe['PercentageMaxLoad'] = percentage_max_loads

    return dataframe


def reorder_dataframe_columns(dataframe):
    dataframe = move_coordinates_to_start(dataframe)
    dataframe = move_percentage_to_end(dataframe)

    return dataframe


def count_coordinate_columns(dataframe):
    return extract_coordinate_data(dataframe).shape[1]


def count_coordinate_dimensions(dataframe):
    coordinate_columns = list(extract_coordinate_data(dataframe))
    dimensions = set(column[-1] for column in coordinate_columns)

    return len(dimensions)