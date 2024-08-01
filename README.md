# Hybrid-Machine-leaning-and-Hydrological-model-for-infiltration-rate-prediction

Instructions for Using the Code to Derive and Predict Infiltration Model Parameters
Overview

This guide provides step-by-step instructions on how to use the provided Python code to:
- Derive predictor variables and infiltration model parameters (for example: Sorptivity and Transmissivity).
- Predict model parameters using Artificial Neural Networks (ANN) and MissForest.
- 
Why Use This Code
1. Derive Key Parameters: The code calculates Sorptivity (S) and Transmissivity (A) from lab-derived soil properties and field infiltration rates.
2. Predict Model Parameters: By using ANN and MissForest, you can predict model parameters at a target test point, enhancing your ability to analyze and interpret infiltration rates.

Steps to Derive Predictor Variables and Model Parameters
1. Derive Parameters from Infiltration Rates
This step calculates Sorptivity (S) and Transmissivity (A) for each station.

Prepare Data: Ensure your Excel file contains columns for time and infiltration rates for each station.
Run the Code:
Replace file_path with the path to your Excel file.
Execute the script to calculate Sorptivity and Transmissivity for each station.
The script prints the calculated parameters for each station.
code:
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
file_path = input("Enter the file path for Philips parameters Excel file: ")
data = pd.read_excel(file_path)

# Extract station names dynamically based on pattern
stations = [col for col in data.columns if 'H' in col or 'K' in col or 'G' in col]
time_columns = [col for col in data.columns if 'Time' in col]

# Initialize lists to store parameters
parameters = []

# Calculate parameters for each station
for i, station in enumerate(stations):
    time_col = time_columns[i]
    x = data[time_col].dropna()
    y = data[station].dropna()

    # Ensure x and y have the same length
    x, y = x[:len(y)], y[:len(x)]

    # Reshape for sklearn
    x_reshaped = np.array(x).reshape(-1, 1)
    y_reshaped = np.array(y).reshape(-1, 1)

    # Linear regression
    model = LinearRegression()
    model.fit(x_reshaped, y_reshaped)
    y_pred = model.predict(x_reshaped)

    # Calculate parameters
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    s = abs(slope) * 2  # Sorptivity
    A = abs(intercept)  # Transmissivity factor

    # Append parameters to the list
    parameters.append({'Station': station, 'Sorptivity (S)': s, 'Transmissivity (A)': A})

# Convert the parameters list to a DataFrame and print it
parameters_df = pd.DataFrame(parameters)
print(parameters_df)

2. Predict Model Parameters Using ANN
This step uses Artificial Neural Networks (ANN) to predict Sorptivity (S) and Transmissivity (A) at a target test point.

Prepare Data: Ensure your Excel file contains all required predictor variables and the target test point.
Run the Code:
Replace file_path with the path to your Excel file.
Execute the script to train the ANN model and predict the parameters for the target test point.
The script prints the predicted values for Sorptivity (S) and Transmissivity (A).
code: from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Load the Excel sheet
file_path = input("Enter the file path for the helper Excel file: ")
df = pd.read_excel(file_path)

# Separate features and target variable
X_train = df.iloc[:-1, :-2]  # All rows except the last one for training
y_train = df.iloc[:-1, -2:]   # Target variables for training, only the last 2 columns
X_test_last_row = df.iloc[-1:, :-2]  # Last row for testing
y_test_last_row = df.iloc[-1:, -2:]  # Actual values for the last 2 columns of the last row

# Define parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
}

# Initialize MLPRegressor
ann_regressor = MLPRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=ann_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and its parameters
best_estimator = grid_search.best_estimator_

# Predict the values for the last 2 columns of the last row using the best estimator
y_test_last_row_predicted = best_estimator.predict(X_test_last_row)

# Print the predicted values for the last 2 columns of the last row using the best estimator
print("Best predicted values for the last 2 columns of the last row:")
print(y_test_last_row_predicted)

# Display the best parameters found
print("\nBest parameters found:")
print(grid_search.best_params_)
print("Mean Squared Error corresponding to the best parameters:", -grid_search.best_score_)

3. Predict Model Parameters Using MissForest
This step uses the MissForest algorithm to predict Sorptivity (S) and Transmissivity (A) at a target test point.

Prepare Data: Ensure your Excel file contains all required predictor variables and the target test point.
Run the Code:
Replace file_path with the path to your Excel file.
Execute the script to impute the missing values and predict the parameters for the target test point.
The script prints the imputed values for Sorptivity (S) and Transmissivity (A).

import pandas as pd
from missingpy import MissForest
from sklearn.metrics import mean_squared_error

# Load the Excel sheet
file_path = input("Enter the file path for the helper Excel file: ")
df = pd.read_excel(file_path)

# Separate features and target variable
X_train = df.iloc[:-1, :-2]  # All rows except the last one for training
y_train = df.iloc[:-1, -2:]   # Target variables for training, only the last 2 columns
X_test_last_row = df.iloc[-1:, :-2]  # Last row for testing
y_test_last_row = df.iloc[-1:, -2:]  # Actual values for the last 2 columns of the last row

# Combine features and target variables for training
train_data = pd.concat([X_train, y_train], axis=1)

# Initialize MissForest
imputer = MissForest()

# Fit MissForest on the training data and transform the last row
imputed_train_data = imputer.fit_transform(train_data)
X_train_imputed = imputed_train_data[:, :-2]
y_train_imputed = imputed_train_data[:, -2:]

# Impute the missing values for the last row
test_data = pd.concat([X_test_last_row, y_test_last_row], axis=1)
imputed_test_data = imputer.transform(test_data)
X_test_last_row_imputed = imputed_test_data[:, :-2]
y_test_last_row_imputed = imputed_test_data[:, -2:]

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_last_row, y_test_last_row_imputed)
print(f"Mean Squared Error for the imputed values: {mse}")

# Print the imputed values for the last 2 columns of the last row
print("Imputed values for the last 2 columns of the last row:")
print(y_test_last_row_imputed)

Conclusion
By following these steps, you can derive key infiltration model parameters and use machine learning techniques to predict these parameters at a target test point. This process helps in accurately modeling infiltration rates and understanding the underlying soil properties and hydrological processes.

Final Step: Fitting the Imputed Infiltration Model Parameters Back to Hydrological Models
Overview
After predicting and imputing the infiltration model parameters (Sorptivity (S) and Transmissivity (A)), the next step is to fit these parameters back into hydrological models to obtain the infiltration rate at the target point.

Why This Step is Important
Validation: This step validates the accuracy and relevance of the predicted parameters by comparing the predicted infiltration rates with observed values.
Application: It allows the use of predicted parameters in practical hydrological models, facilitating water resource management and planning.

import numpy as np

# Function to calculate cumulative infiltration using Philip's model
def cumulative_infiltration(t, S, A):
    return S * np.sqrt(t) + A * t

# Function to calculate infiltration rate using Philip's model
def infiltration_rate(t, S, A):
    return S / (2 * np.sqrt(t)) + A

# Example usage
# Assuming S and A are the predicted parameters for the target point
S_pred = y_test_last_row_imputed[0][0]
A_pred = y_test_last_row_imputed[0][1]

# Time points for which to calculate infiltration (e.g., every hour up to 10 hours)
time_points = np.arange(1, 11)  # 1 to 10 hours

# Calculate cumulative infiltration and infiltration rate at each time point
cumulative_infiltrations = cumulative_infiltration(time_points, S_pred, A_pred)
infiltration_rates = infiltration_rate(time_points, S_pred, A_pred)

# Display the results
results = pd.DataFrame({
    'Time (hours)': time_points,
    'Cumulative Infiltration (mm)': cumulative_infiltrations,
    'Infiltration Rate (mm/hr)': infiltration_rates
})

print(results)
