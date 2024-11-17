import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../datasets/data.csv')

# Separate the features and target variable
X = data.drop(columns=['Target'])
y = data['Target']

# Optional: Discretize continuous features like Age
# X['Age_binned'] = pd.cut(X['Age'], bins=[-np.inf, 18, 30, 45, 60, np.inf], labels=['<18', '18-30', '30-45', '45-60', '>60'])
# X = X.drop(columns=['Age'])  # Remove the original Age column to avoid redundancy

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# One-hot encode categorical features (if any exist due to binning or other preprocessing)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns of training and testing sets (to handle possible column mismatches after one-hot encoding)
X_train, X_test = X_train.align(X_test, axis=1, fill_value=0)

# Create a Decision Tree Classifier
cart_model = DecisionTreeClassifier(random_state=42)

# Set up the hyperparameter grid
param_grid = {
    'criterion': ['entropy'],
    'max_depth': [7],  # Limit depth to avoid overfitting
    'min_samples_split': [40],  # Increase minimum samples required to split a node
    'min_samples_leaf': [40],  # Increase minimum samples required in leaf nodes
    'ccp_alpha': [0.10],  # Cost-complexity pruning
    'max_features': [None, 'sqrt', 'log2']  # Experiment with feature selection at splits
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=cart_model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_cart_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions
y_pred = best_cart_model.predict(X_test)

# Calculate statistics
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print statistics
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(class_report)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    best_cart_model, 
    filled=True, 
    feature_names=X_train.columns, 
    class_names=[str(cls) for cls in np.unique(y)], 
    rounded=True
)
plt.title("Decision Tree Visualization")
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Display feature importances
importances = pd.Series(best_cart_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Feature Importances')
plt.show()

# Evaluate Training Accuracy
y_train_pred = best_cart_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

# Evaluate Cross-Validation Mean Accuracy
cv_mean_score = grid_search.best_score_
print(f"Cross-Validation Mean Accuracy: {cv_mean_score}")

# Plot Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_cart_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(12, 8))
plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue', marker='o')
plt.plot(train_sizes, test_scores_mean, label='Validation Score', color='orange', marker='s')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
