import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts


def get_column_names(path):
    '''
        Gets the column names from the given path
    '''
    data = pd.read_csv(path)
    return data.columns


def get_X_y(path):
    """
    Loads the X and y from the given path.
    Assumes last columns of the x are the target values.
    :param path: the path to the x
    :return: the x as X and y numpy arrays
    """
    x = pd.read_csv(path)
    X = x.drop(x.columns[-1], axis=1).to_numpy()
    y = x[x.columns[-1]].to_numpy()
    return X, y


def train_test_split(X, y, train_size, shuffle=True, seed=42):
    """
    Splits the x into training and test sets.
    :param X: the x
    :param y: the target values
    :param train_size: the size of the training set
    :param shuffle: whether to shuffle the x
    :param seed: the seed for the random generator
    :return: X_train, X_test, y_train, y_test
    """
    # length = len(X)
    # n_train = int(np.ceil(length*train_size))
    # n_test = length - n_train

    # if shuffle:
    #     perm = np.random.RandomState(seed).permutation(length)
    #     test_indices = perm[:n_test]
    #     train_indices = perm[n_test:]
    # else:
    #     train_indices = np.arange(n_train)
    #     test_indices = np.arange(n_train, length)

    # X_train = X[train_indices]
    # X_test = X[test_indices]
    # y_train = y[train_indices]
    # y_test = y[test_indices]
    X_train, X_test, y_train, y_test = tts(
        X, y, stratify=y, test_size=1-train_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def train_val_test_split(X, y, train_size, val_size, shuffle=True, seed=42):
    '''
    Splits the x into training, validation and test sets.
    :param X: the x
    :param y: the target values
    :param train_size: the size of the training set
    :param val_size: the size of the validation set
    :param shuffle: whether to shuffle the x
    :param seed: the seed for the random generator
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    '''
    # length = len(X)
    # n_train = int(np.ceil(length*train_size))
    # n_val = int(np.ceil(length*val_size))
    # n_test = length - n_train - n_val

    # if shuffle:
    #     perm = np.random.RandomState(seed).permutation(length)
    #     test_indices = perm[:n_test]
    #     val_indices = perm[n_test:n_test+n_val]
    #     train_indices = perm[n_test+n_val:]
    # else:
    #     train_indices = np.arange(n_train)
    #     val_indices = np.arange(n_train, n_train + n_val)
    #     test_indices = np.arange(n_train + n_val, length)

    # X_train = X[train_indices]
    # X_val = X[val_indices]
    # X_test = X[test_indices]
    # y_train = y[train_indices]
    # y_val = y[val_indices]
    # y_test = y[test_indices]

    test_size = 1-train_size-val_size
    X_train, X_temp, y_train, y_temp = tts(
        X, y, stratify=y, test_size=(1.0 - train_size), random_state=seed)
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = tts(
        X_temp, y_temp, stratify=y_temp, test_size=relative_test_size, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def check_purity(y):
    """
    Checks if the given array is pure.
    :param y: the array
    :return: True if the array is pure, False otherwise
    """
    return len(set(y)) == 1


def classify_array(y):
    """
    Classifies the array into a single class.
    find most common number and return that
    :param y: the array
    :return: the class
    """
    classes, counts = np.unique(y, return_counts=True)
    return classes[counts.argmax()]
    # return np.argmax(np.bincount(y.astype(int)))


def get_possible_breaks(X, type_arr):
    '''
        Calculates possible breaks for a given set of features 
    '''
    breaks = {}
    for col_idx in range(X.shape[1]):
        unique_vals = np.unique(X[:, col_idx])
        num_vals = np.unique(X[:, col_idx]).shape[0]

        type = type_arr[col_idx]
        if type == "cont":
            breaks[col_idx] = []
            for i in range(1, num_vals):
                current_value = unique_vals[i]
                previous_value = unique_vals[i - 1]
                potential_split = (current_value + previous_value) / 2
                breaks[col_idx].append(potential_split)
        elif num_vals > 1:
            breaks[col_idx] = unique_vals
    return breaks


def create_children_np(X, y, col_idx, col_val, type_arr):
    '''
        Creates the children of a dataset given split column and value
    '''
    y = y.reshape(-1, 1)
    X_n = np.hstack((X, y))
    relevant_column = X_n[:, col_idx]
    # print(relevant_column)
    # print(relevant_column<=col_val)
    if type_arr[col_idx] == "cont":
        X_one = X_n[relevant_column <= col_val]
        X_two = X_n[relevant_column > col_val]
    else:
        X_one = X_n[relevant_column == col_val]
        X_two = X_n[relevant_column != col_val]

    # print(X_one.shape, X_two.shape)
    Y_one = X_one[:, -1]
    Y_two = X_two[:, -1]
    X_one = X_one[:, :-1]
    X_two = X_two[:, :-1]

    # print(X_one.shape, X_two.shape, Y_one.shape, Y_two.shape)
    return X_one, Y_one, X_two, Y_two


def calc_entropy_np(y):
    """
    Calculates the entropy of the given array.
    :param y: the array
    :return: the entropy
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return np.sum(probabilities * -np.log2(probabilities))


def calc_info_gain(X, y, col_idx, col_val, type_arr):
    '''
        Calculates the information gain of a given split
    '''
    X_one, Y_one, X_two, Y_two = create_children_np(
        X, y, col_idx, col_val, type_arr)
    p = len(X_one) / len(X)
    return calc_entropy_np(y) - (p * calc_entropy_np(Y_one) + (1 - p) * calc_entropy_np(Y_two))


def calc_gini_np(y):
    """
    Calculates the gini impurity of the given array.
    :param y: the array
    :return: the gini impurity
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)


def calc_gini_gain(X, y, col_idx, col_val, type_arr):
    '''
        Calculates the gini index of a given split
    '''
    X_one, Y_one, X_two, Y_two = create_children_np(
        X, y, col_idx, col_val, type_arr)
    p = len(X_one) / len(X)
    return calc_gini_np(y) - (p * calc_gini_np(Y_one) + (1 - p) * calc_gini_np(Y_two))


def get_best_split(X, y, type_arr, method="entropy"):
    '''
        Calculates the best split for a given set of features
    '''
    best_col = -1
    best_val = -1
    best_gain = -10000
    breaks = get_possible_breaks(X, type_arr)
    for col_idx in breaks:
        for col_val in breaks[col_idx]:
            if method == "entropy":
                gain = calc_info_gain(X, y, col_idx, col_val, type_arr)
            else:
                gain = calc_gini_gain(X, y, col_idx, col_val, type_arr)
            if gain > best_gain:
                best_col = col_idx
                best_val = col_val
                best_gain = gain
    return best_col, best_val


def assign_feature_type(X, cont_thresh):
    '''
        Assigns the type of each feature based on the data
    '''
    type_arr = []
    for col_idx in range(X.shape[1]):
        type_val = X[:, col_idx][0]
        unique_vals = np.unique(X[:, col_idx])
        if len(unique_vals) < cont_thresh or isinstance(type_val, str):
            type_arr.append("discrete")
        else:
            type_arr.append("cont")
    return type_arr


def calc_accuracy(y_true, y_pred):
    '''
        Calculates the accuracy of the prediction
    '''
    return np.sum(y_pred == y_true) / len(y_pred)


def filter(X, y, col_idx, col_val, type_arr):

    y = y.reshape(-1, 1)
    X_n = np.hstack((X, y))
    relevant_column = X_n[:, col_idx]

    if type_arr[col_idx] == "cont":
        X_yes = X_n[relevant_column <= col_val]
        X_no = X_n[relevant_column > col_val]

    else:
        X_yes = X_n[relevant_column == col_val]
        X_no = X_n[relevant_column != col_val]

    Y_yes = X_yes[:, -1]
    Y_no = X_no[:, -1]
    X_yes = X_yes[:, :-1]
    X_no = X_no[:, :-1]
    return X_yes, Y_yes, X_no, Y_no


def check_node(X, y):
    return classify_array(y)
