#*******************************************************************************
# File Name: main.py
# Course: Comp 3625 - Artificial Intelligence
# Assignment: 2
# Due Date: 2024-11-7
# Made By: Clency Tabe & Glenn Yeap
#********************************************************************************


import pickle
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Constants for cost and percentage gain from successful contacts
contact_cost = 10
percentage_gain = 0.04

def identify_customers(data: pd.DataFrame):
    """
    This function identifies the customers that the model predicts as successful contacts.
    It calculates the revenue, costs, and profit for the predicted contacts.

    Args:
        data (pd.DataFrame): The input dataset including features and the target 'success' column.

    Returns:
        np.ndarray: Predictions of customer success (1 for success, 0 for failure).
    """
    # Splitting data into features (X) and target variable (y)
    x = data.drop(columns='success')
    y = data['success']

    # test on a subset of the data
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=3)

    # Load the pre-trained model from file
    with open('./model.pkl', 'rb') as file:  # rb = binary read mode
        model = pickle.load(file)

    # Predict with the pre-trained model
    pred = model.predict(x)

    # Output confusion matrix and classification report
    cm = confusion_matrix(y, pred)
    print("\n\nConfusion Matrix:")
    print(cm)

    cr = classification_report(y, pred)
    print("Classification Report:")
    print(cr)

    # Calculate revenue from successful contacts
    balance = x['balance'].values
    successful_contacts = (pred == 1) & (y == 1)
    total_revenue = np.sum(balance[successful_contacts] * percentage_gain)

    # Calculate the number of successful contacts and their corresponding cost
    num_successful_contacts = np.sum(successful_contacts)

    # Output results
    print(f"Total contacts: {np.sum(pred)}")
    print(f"Total cost of contacts: ${np.sum(pred) * contact_cost}")
    print(f"Total successful contacts: {num_successful_contacts}")
    print(f"Total Revenue: ${total_revenue:.2f}")
    print(f"Total Profit: ${total_revenue - (np.sum(pred) * contact_cost):.2f}")
    print(f"Wasted costs: ${(np.sum(pred) * contact_cost) - num_successful_contacts :.2f}")

    return pred


def show_data(data: pd.DataFrame):
    """
    Displays basic information about the dataset, including its shape, descriptions,
    and value counts for the target 'success' variable.
    """
    print("-------------------------------------------------------------------------------")
    print(f'\n + {data.head()}')
    print(data.describe())
    print(data.info())
    print("Shape of the data: ", data.shape)
    print('Class Categories', data['success'].unique())
    print(f"{Counter(data['success'])}")
    print("-------------------------------------------------------------------------------")


def visualize_data(data: pd.DataFrame):
    """
    Visualizes the correlation between features and the target variable 'success'.
    """
    print(f"Imbalance: {Counter(data['success'])}")

    # Visualizing correlation with the target variable 'success'
    m = data.corr()['success']
    m.plot.bar(figsize=(16, 9), title='Correlation with the Success Variable', grid=True)
    plt.show()


def under_over_sample(x, y, sample_type: int):
    """
    Applies over-sampling or under-sampling based on the specified sample_type.
    0 for under-sampling, 1 for over-sampling.

    Args:
        x (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        sample_type (int): Sampling method (0 for under-sampling, 1 for over-sampling).

    Returns:
        tuple: The resampled feature data (x_sam) and target data (y_sam).
    """
    if sample_type == 0:
        # Under-sampling with RandomUnderSampler
        sampler = RandomUnderSampler(random_state=42)
    else:
        # Over-sampling with SMOTE (Synthetic Minority Over-sampling Technique)
        sampler = SMOTE(random_state=42)

    return sampler.fit_resample(x, y)


def random_forest(x, y):
    """
    Trains a RandomForest model with over-sampling, evaluates it, and saves the model.

    Args:
        x (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
    """
    # Apply over-sampling to balance the dataset
    x, y = under_over_sample(x, y, 1)

    # Train-test split with stratification to preserve class distribution
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=3)

    # Define and train the RandomForest model with optimized hyperparameters
    model = RandomForestClassifier(n_estimators=300, max_depth=50, max_features='sqrt',
                                   min_samples_leaf=1, min_samples_split=2, random_state=42)
    model.fit(x_train, y_train)

    # Save the trained model to a file for future use
    with open('./model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Evaluate the model on the test set
    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Classification report and confusion matrix
    cr = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(cr)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    # Load the dataset
    dataset = pd.read_csv('./data/3625_assign2_data_train.csv')
    print(f'Counter: {Counter(dataset["success"])}')

    # Splitting dataset into features (X) and target variable (y)
    x = dataset.drop(columns='success')
    y = dataset['success']

    # Train the RandomForest model (commented out here, use if needed)
    # random_forest(x, y)

    # Run the prediction for customer success and calculate profit
    prediction = identify_customers(dataset)
