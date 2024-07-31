#make excel file with pridictor variables
#the pridictor parameters like " % gravel  % sand  % Silt  %water cont" are derived from lab analysis and for Phililp's model 
#(for example) the S (Sorptivity) and A (Transmissivity factor) are derived using infiltration rates of different test points and use the following code to derive those parameters


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

###after deriving the pridictor variables and creating the excel sheet make a test point as a target and use the below 
#the excel file should have row with parameters "Test point	% of gravel S	% of sand S	% of silt S	water cont S	% of gravel 0.5	% of sand 0.5	% of silt 0.5	water cont 0.5	% of gravel 1	% of sand 1	% of silt 1	water cont 1	S	K"
###ANN code to predict the model parameters here for example "S" and "A"

from sklearn.experimental import enable_iterative_imputer  # noqa
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

#use the predicted model's parameters and fit the infiltration model for infiltration rates at target point
#example: same way after deriving the predictor variables as above the same excel sheet can use fed to MissForest to predict model parameters at target point
### use the MissForest code to do so

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

# Print the imputed values for the last row
print("Imputed values for the last 2 columns of the last row:")
print(y_test_last_row_imputed)

#use the predicted model's parameters and fit the infiltration model for infiltration rates at target point

