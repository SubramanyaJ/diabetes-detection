import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../datasets/data.csv')

# Separate the features and target variable
X = data.drop(columns=['Target'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual classifiers
naive_bayes = GaussianNB()
decision_tree = DecisionTreeClassifier(random_state=40)
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
log_reg_l2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=42, max_iter=1000)

# Parameter grids for GridSearchCV
param_grid_nb = {'var_smoothing': [1e-8]}
param_grid_dt = {'criterion': ['entropy'], 'splitter': ['best'], 'max_depth': [7], 'min_samples_split': [80], 'min_samples_leaf': [50], 'max_features': [None]}
param_grid_logreg = {'C': [0.1, 1, 10]}

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[('naive_bayes', naive_bayes), ('decision_tree', decision_tree), ('log_reg_l1', log_reg_l1), ('log_reg_l2', log_reg_l2)],
    voting='soft',
    weights=[1, 5, 2, 2]
)

# Parameter grid for voting classifier
param_grid_voting = {
    'naive_bayes__var_smoothing': param_grid_nb['var_smoothing'],
    'decision_tree__criterion': param_grid_dt['criterion'],
    'decision_tree__splitter': param_grid_dt['splitter'],
    'decision_tree__max_depth': param_grid_dt['max_depth'],
    'decision_tree__min_samples_split': param_grid_dt['min_samples_split'],
    'decision_tree__min_samples_leaf': param_grid_dt['min_samples_leaf'],
    'log_reg_l1__C': param_grid_logreg['C'],
    'log_reg_l2__C': param_grid_logreg['C']
}

# Cross-validation setup with StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=voting_clf, param_grid=param_grid_voting, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test_scaled)

# Calculate statistics
accuracy = np.mean(y_pred == y_test)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print statistics
print(f"\nBest Hyperparameters: {grid_search.best_params_}")
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:")
print(class_report)

# Visualize the Decision Tree in the best model
best_decision_tree = best_model.named_estimators_['decision_tree']

plt.figure(figsize=(80, 60))  # Increase figure size
plot_tree(
    best_decision_tree,
    filled=True,
    feature_names=X.columns,
    class_names=[str(cls) for cls in np.unique(y)],
    rounded=True,
    fontsize=12  # Increase font size
)
plt.title("Decision Tree in Voting Classifier")
plt.show()


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



