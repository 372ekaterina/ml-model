import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)


# Group by Pclass and Sex and calculate the median age for each group
grouped = df_train.groupby(['Pclass', 'Sex'])['Age'].median()

# fill_age function to fill missing values of Age
def fill_age(row):
    if pd.isnull(row['Age']):
        return grouped[row['Pclass'], row['Sex']]
    else:
        return row['Age']

# Apply the function to fill missing ages
df_train['Age'] = df_train.apply(fill_age, axis=1)
df_test['Age'] = df_test.apply(fill_age, axis=1)


# MODEL
# Separate the features 
y = data_train['Survived']
feature_columns = ['Pclass', 'Sex', 'Age']
X = data_train[feature_columns]

# Split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)

# Specify the model
titanic_model = DecisionTreeRegressor(random_state=1)

# Fit model with the training data.
titanic_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = titanic_model.predict(val_X)

# Calculate the Mean Absolute Error in Validation Data
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

# Save CSV file
val_predictions.to_csv('data/best_model_submission.csv', index=False)
