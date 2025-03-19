import pandas as pd
import yaml
from data_loader import read_raw
from validation_split import data_split


def process_data(data_config):
    """ Process raw data into a format that can be used by the model
    :param data_config: dict, configuration parameters for data processing
    :return: None
    """

    x, y, problem_ids, student_ids = [], [], [], []
    # concatenate all problems together
    for problem_id, raw_filepath in enumerate(data_config['raw_paths']):
        problem_x, problem_y, problem_problem_ids, problem_student_ids = read_raw(raw_filepath, data_config, problem_id)
        x.extend(problem_x)
        y.extend(problem_y)
        problem_ids.extend(problem_problem_ids)
        student_ids.extend(problem_student_ids)

    x = pd.Series(x)
    y = pd.Series(y)
    problem_ids = pd.Series(problem_ids)
    student_ids = pd.Series(student_ids)

    data_split(x, y, problem_ids, student_ids, data_config)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process raw data into a format that can be used by the model')
    parser.add_argument('--data_config', type=str, help='path to data configuration file')
    args = parser.parse_args()

    # load yaml from path
    data_config = yaml.load(open(args.data_config), Loader=yaml.FullLoader)

    process_data(data_config)
