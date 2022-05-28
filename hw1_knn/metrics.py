import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    
    true_pos = np.size(y_pred[np.where(np.logical_and(y_true=='1', y_pred=='1'))])
    true_pos_and_false_pos = np.size(y_pred[np.where(y_pred=='1')])
    true_pos_and_false_neg = np.size(y_pred[np.where(y_true=='1')])
    
    
    if true_pos_and_false_pos == 0:
        precision = float("inf")
    else:
        precision = true_pos / true_pos_and_false_pos
    if true_pos_and_false_neg == 0:
        recall = float("inf")
    else:
        recall = true_pos / true_pos_and_false_neg
    
    if true_pos == 0:
        f1 = float("inf")
    elif true_pos_and_false_pos == 0 or true_pos_and_false_neg == 0:
        f1 = 0
    else: 
        f1 = 2 / (recall ** (-1) + precision ** (-1))
    
    accuracy = (y_true == y_pred).mean()   
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = (y_true == y_pred).mean()   
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    
    if np.sum((y_true - np.mean(y_true)) ** 2) == 0:
        r2 = - float("inf")
    else:
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.sum((y_true - y_pred) ** 2) / np.size(y_true)
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.sum(abs(y_true - y_pred)) / np.size(y_true)
    
    return mae
    