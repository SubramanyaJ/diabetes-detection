import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    data = pd.read_csv('/home/smgb/College/project/diabetes_detection/datasets/data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file not found. Ensure 'data.csv' is in the correct path.")
    exit()

# Separate the features and target variable
try:
    X = data.drop(columns=['Target'])
    y = data['Target']
    print("Features and target variable separated.")
except KeyError:
    print("Error: 'Target' column not found in dataset.")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split into training and testing sets.")

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

# Create a RandomForest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Set up the hyperparameter grid for RandomForest
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'criterion': ['gini', 'entropy'],  # Criteria for splitting
    'max_depth': [10, 12, None],  # Max depth of the trees
    'min_samples_split': [2, 5],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2],  # Minimum samples required in a leaf
    'bootstrap': [True, False],  # Whether bootstrap samples are used
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)

# Train the model with hyperparameter tuning
print("Starting model training...")
grid_search.fit(X_train_scaled, y_train)
print("Model training complete.")

# Get the best model and its parameters
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Make predictions on the test data
y_pred = best_rf_model.predict(X_test_scaled)
print("Predictions made on test data.")

# Calculate statistics
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print out the statistics
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("Confusion matrix plotted.")