import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../datasets/data.csv')

# Separate the features and target variable
X = data.drop(columns=['Target'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes parameters (accessible for tuning)
naive_bayes_params = {
    'var_smoothing': 1e-8  # Default value
}

# Decision Tree parameters (accessible for tuning)
decision_tree_params = {
    'criterion': 'entropy',         # 'gini' or 'entropy'
    'splitter': 'best',          # 'best' or 'random'
    'max_depth': 9,           # Maximum depth of the tree
    'min_samples_split': 20,      # Minimum number of samples required to split an internal node
    'min_samples_leaf': 10,       # Minimum number of samples required to be a leaf node
    'max_features': None,        # The number of features to consider when looking for the best split
    'random_state': 40          # Ensures reproducibility
}

# Create individual classifiers with tunable hyperparameters
naive_bayes = GaussianNB(**naive_bayes_params)
decision_tree = DecisionTreeClassifier(**decision_tree_params)

# Create a voting classifier that combines both models
voting_clf = VotingClassifier(
    estimators=[('naive_bayes', naive_bayes), ('decision_tree', decision_tree)],
    voting='soft',  # Use 'soft' for probability-based voting
    weights = [1,  5]
)

# Train the voting classifier
voting_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test_scaled)

# Calculate relevant statistics
accuracy = np.mean(y_pred == y_test)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print statistics
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

