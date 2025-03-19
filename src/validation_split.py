import numpy as np
import pandas as pd
import os

def data_split(x, y, problem_ids, student_ids, data_config):
    """
    Split data into train, val, and test sets
    :param x: np.array, student responses
    :param y: np.array, labels
    :param problem_ids: np.array, problem ids
    :param student_ids: np.array, student ids
    :param data_config: dict, data configuration
    :return: list of tuples, [(train, val, test), ...]
    """
    num_val_students = data_config["num_val_students"]
    num_test_students = data_config["num_test_students"]
    num_folds = data_config["num_folds"]
    seed = data_config["seed"]
    # get unique student ids
    unique_student_ids = np.unique(student_ids)

    prng = np.random.default_rng(seed)

    # shuffle student ids
    prng.shuffle(unique_student_ids)
    # select num_test_students student ids for test set
    # but make sure that each test student has a response to all 4 problems
    test_student_ids = []
    for student_id in unique_student_ids:
        student_problem_ids = problem_ids[student_ids == student_id]
        if len(np.unique(student_problem_ids)) == 4:
            test_student_ids.append(student_id)
        if len(test_student_ids) == num_test_students:
            break
    test_student_ids = np.array(test_student_ids)

    # create a new list of unique_student_ids with the test_student_ids at the start
    remaining_student_ids = np.setdiff1d(unique_student_ids, test_student_ids)
    unique_student_ids = np.concatenate((test_student_ids, remaining_student_ids))
    val_student_ids = unique_student_ids[num_test_students:num_test_students +num_val_students]
    train_student_ids = unique_student_ids[num_test_students + num_val_students:]

    # assert that train val and test are all different and include all students
    assert len(np.intersect1d(train_student_ids, val_student_ids)) == 0
    assert len(np.intersect1d(train_student_ids, test_student_ids)) == 0
    assert len(np.intersect1d(val_student_ids, test_student_ids)) == 0
    assert len(np.union1d(np.union1d(train_student_ids, val_student_ids), test_student_ids)) == len(unique_student_ids)
    
    # Exclude test student ids from the list of unique student ids
    remaining_student_ids = np.setdiff1d(unique_student_ids, test_student_ids)
    
    # Calculate the number of validation students per fold
    num_val_students_per_fold = len(remaining_student_ids) // num_folds

    for fold in range(num_folds):
        fold_val_student_ids = remaining_student_ids[num_val_students_per_fold * fold:num_val_students_per_fold * (fold + 1)]
        fold_train_student_ids = np.setdiff1d(remaining_student_ids, fold_val_student_ids)

        # assert that train val and test are all different and include all students
        assert len(np.intersect1d(fold_train_student_ids, fold_val_student_ids)) == 0
        assert len(np.intersect1d(fold_train_student_ids, test_student_ids)) == 0
        assert len(np.intersect1d(fold_val_student_ids, test_student_ids)) == 0
        assert len(np.union1d(np.union1d(fold_train_student_ids, fold_val_student_ids), test_student_ids)) == len(unique_student_ids)

        fold_train_indices = np.where(np.isin(student_ids, fold_train_student_ids))[0]
        fold_val_indices = np.where(np.isin(student_ids, fold_val_student_ids))[0]

        fold_train = (x[fold_train_indices], y[fold_train_indices], problem_ids[fold_train_indices], student_ids[fold_train_indices])
        fold_val = (x[fold_val_indices], y[fold_val_indices], problem_ids[fold_val_indices], student_ids[fold_val_indices])

        # concatenate problem_ids and student_ids to x then save x and y
        fold_train_x = pd.concat((fold_train[0], fold_train[2], fold_train[3]), axis=1)  
        fold_val_x = pd.concat((fold_val[0], fold_val[2], fold_val[3]), axis=1)  

        # save x and y
        file_path = data_config["folded_train_x_file"].format(fold=fold, num_test_students=num_test_students)  # Format the path with the fold number
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directories if they don't exist

        pd.DataFrame(fold_train_x).to_csv(data_config["folded_train_x_file"].format(fold=fold, num_test_students=num_test_students), index=False)
        pd.DataFrame(fold_train[1]).to_csv(data_config["folded_train_y_file"].format(fold=fold, num_test_students=num_test_students), index=False)
        pd.DataFrame(fold_val_x).to_csv(data_config["folded_val_x_file"].format(fold=fold, num_test_students=num_test_students), index=False)
        pd.DataFrame(fold_val[1]).to_csv(data_config["folded_val_y_file"].format(fold=fold, num_test_students=num_test_students), index=False)


    # split data into train, val, and test sets
    train_indices = np.where(np.isin(student_ids, train_student_ids))[0]
    val_indices = np.where(np.isin(student_ids, val_student_ids))[0]
    test_indices = np.where(np.isin(student_ids, test_student_ids))[0]

    train = (x[train_indices], y[train_indices], problem_ids[train_indices], student_ids[train_indices])
    val = (x[val_indices], y[val_indices], problem_ids[val_indices], student_ids[val_indices])
    test = (x[test_indices], y[test_indices], problem_ids[test_indices], student_ids[test_indices])
    retrain_indices = np.concatenate((train_indices, val_indices))
    retrain = (x[retrain_indices], y[retrain_indices], problem_ids[retrain_indices], student_ids[retrain_indices])

    # concatenate problem_ids and student_ids to x then save x and y
    train_x = pd.concat((train[0], train[2], train[3]), axis=1)  
    val_x = pd.concat((val[0], val[2], val[3]), axis=1)  
    rertain_x = pd.concat((retrain[0], retrain[2], retrain[3]), axis=1)
    test_x = pd.concat((test[0], test[2], test[3]), axis=1)  

    # save x and y
    pd.DataFrame(train_x).to_csv(data_config["train_x_file"].format(num_test_students=num_test_students), index=False)
    pd.DataFrame(train[1]).to_csv(data_config["train_y_file"].format(num_test_students=num_test_students), index=False)
    pd.DataFrame(val_x).to_csv(data_config["val_x_file"].format(num_test_students=num_test_students), index=False)
    pd.DataFrame(val[1]).to_csv(data_config["val_y_file"].format(num_test_students=num_test_students), index=False)
    pd.DataFrame(rertain_x).to_csv(data_config["retrain_x_file"].format(num_test_students=num_test_students), index=False)
    pd.DataFrame(retrain[1]).to_csv(data_config["retrain_y_file"].format(num_test_students=num_test_students), index=False)
    pd.DataFrame(test_x).to_csv(data_config["test_x_file"].format(num_test_students=num_test_students), index=False)
    pd.DataFrame(test[1]).to_csv(data_config["test_y_file"].format(num_test_students=num_test_students), index=False)


    return
