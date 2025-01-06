# /**
#   *   Uses the Random Forest classifier for diabetes prediction.
#   *   Accuracy : 96.672%
#   *   Optimal Parameters :
#   *   Maximum Leaf Nodes : 25
#   *   Tree Estimators : 96
#   *   Maximum Tree Depth : 6
#   *   
#   *   
#   **/


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = '../../datasets/data.csv'  # Replace with your file path
data = pd.read_csv(data_path)

# Feature-target separation
X = data.drop(columns=['Target'])
y = data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define a Bayesian Random Forest (using a base RandomForestClassifier for simplicity)
rf = RandomForestClassifier(max_leaf_nodes=25, random_state=42)

# Perform GridSearchCV to find optimal parameters
param_grid = {
    'n_estimators': [96],
    'max_depth': [6],
    'min_samples_split': [100],
    'min_samples_leaf': [80],
    'bootstrap' : [True],
    'max_features' : ['sqrt'],
    'criterion' : ['entropy'],
#   'ccp_alpha': [0.2]
    'class_weight': ['balanced_subsample']
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           #funnel plot
                           scoring='recall', cv=8, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Perform cost-complexity pruning
importances = best_rf.feature_importances_
important_features = np.argsort(importances)[-len(importances)//2:]
X_train_pruned = X_train #.iloc[:, important_features]
X_test_pruned = X_test #.iloc[:, important_features]

# Cross-validation to check for overfitting
cv_scores = cross_val_score(best_rf, X_train_pruned, y_train, cv=12, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Fit pruned model
best_rf.fit(X_train_pruned, y_train)

# Predictions and confusion matrix
y_pred = best_rf.predict(X_test_pruned)
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
cmd.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot a single random tree from the forest
random_tree = np.random.choice(best_rf.estimators_)
plt.figure(figsize=(20, 10))
plot_tree(random_tree, feature_names=X_train_pruned.columns, class_names=best_rf.classes_.astype(str), filled=True)
plt.title("Random Tree from the Random Forest")
plt.show()

# Funnel plots
# Parameters for type 2 and type 3 distinguish
#! Brain insulin resistance and neurotoxins

# Extract tree depths and corresponding scores
tree_depths = [estimator.tree_.max_depth for estimator in best_rf.estimators_]
tree_scores = [estimator.score(X_test_pruned, y_test) for estimator in best_rf.estimators_]

# Create a DataFrame for visualization
tree_stats = pd.DataFrame({'Tree Depth': tree_depths, 'Score': tree_scores})

# Funnel plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Tree Depth', y='Score', data=tree_stats, alpha=0.7, color='blue')
sns.lineplot(x='Tree Depth', y='Score', data=tree_stats, color='orange', label='Trend')
plt.title("Funnel Plot of Tree Depth vs. Score")
plt.xlabel("Tree Depth")
plt.ylabel("Score")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

