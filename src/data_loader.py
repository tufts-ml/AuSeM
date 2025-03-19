import pandas as pd
import torch
import datasets

def read_raw(file_path, data_config, problem_id):
    """
    Read data from an Excel file
    :param file_path: str, path to the csv file
    :param sheet_name: str, name of the sheet to read
    :return: pd.DataFrame, data read from the csv file
    """
    
    # read csv, converting string list into actual list object
    data = pd.read_csv(file_path)

    y = data[data_config['label_col']].apply(eval)

    x = data[data_config['response_col']]

    problem_ids = [problem_id] * len(data)
    
    student_ids = data[data_config['student_id_col']]

    return x, y, problem_ids, student_ids


def to_device(dataset, device):
    """
    Move a dataset to torch tensors and a device
    :param dataset: tuple, (x,y,problem_id,student_id) dataset
    :param device: torch.device, device to move the data to
    :return: tuple, (X, y) dataset on the device
    """
    x, y, problem_id, student_id = dataset

    x = torch.tensor(x.values).to(device)
    y = torch.tensor(y.values).to(device)
    problem_id = torch.tensor(problem_id.values).to(device)
    student_id = torch.tensor(student_id.values).to(device)

    return torch.utils.data.TensorDataset(x, y, problem_id, student_id)


def create_hf_dataset(tuple_dataset):
    texts, targets, problem_indices, prediction_counts, criteria_texts = tuple_dataset

    dataset = datasets.Dataset.from_dict({
        'text': texts,
        'targets': targets,
        'problem_indices': problem_indices,
        'prediction_counts': prediction_counts,
    })

    return dataset, criteria_texts


def load_datasets(hyper_config, problem_config, train=False, val=False, test=False):
    
    outputs = ()

    if train:
        train_datasets = []
    if val:
        val_datasets = []
    if test:
        test_datasets = []

    for fold in range(hyper_config['num_folds']):
        if train:
            processed_train_features, processed_train_labels = load_processed(hyper_config['train_x_file'].format(fold=fold),
                                                                              hyper_config['train_y_file'].format(fold=fold))
            train_dataset = make_dataset(processed_train_features, processed_train_labels, problem_config)
            train_datasets.append(train_dataset)

        if val:
            processed_val_features, processed_val_labels = load_processed(hyper_config['val_x_file'].format(fold=fold),
                                                                          hyper_config['val_y_file'].format(fold=fold))
            val_dataset = make_dataset(processed_val_features, processed_val_labels, problem_config)
            val_datasets.append(val_dataset)

        if test:
            processed_test_features, processed_test_labels = load_processed(hyper_config['test_x_file'].format(fold=fold), 
                                                                            hyper_config['test_y_file'].format(fold=fold))
            test_dataset = make_dataset(processed_test_features, processed_test_labels, problem_config)
            test_datasets.append(test_dataset)

    if train:
        outputs += (train_datasets,)
    if val:
        outputs += (val_datasets,)
    if test:
        outputs += (test_datasets,)

    return outputs

def load_processed(x_path, y_path):
    # make sure x values are all strings
    x = pd.read_csv(x_path)
    # todo make sure x are nice strings
    y = pd.read_csv(y_path)
    import ast
    import numpy as np
    y['0'] = y['0'].apply(lambda x: list(ast.literal_eval(x)))
    list_of_arrays = y['0'].apply(lambda x: np.array(x)).tolist()

    # Find the maximum length of the arrays
    max_length = max(len(arr) for arr in list_of_arrays)

    # Pad the arrays with NaNs to make them the same length
    padded_y = np.array([np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in list_of_arrays], dtype=np.float32)

    return x, padded_y

def make_dataset(features, labels, problem_config):

    # extract problem_id and student_id as last 2 columns of x
    problem_ids = features.iloc[:, 1].values
    student_ids = features.iloc[:, 2]
    
    # go from Student ## -> ##
    student_ids = student_ids.apply(lambda x: int(x.split(' ')[1])).values

    x = features.iloc[:, 0].values.tolist()
    y = labels
    criteria_questions = {problem_id: problem_config['problems'][problem_id]['questions'] for problem_id in problem_ids}

    # create pytorch dataset
    dataset = (x, y, problem_ids, student_ids, criteria_questions)

    return dataset

