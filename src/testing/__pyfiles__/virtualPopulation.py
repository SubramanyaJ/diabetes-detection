import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'data.csv'
data = pd.read_csv(data_path)

# Feature-target separation
X = data.drop(columns=['Target'])
y = data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

rf = RandomForestClassifier(max_leaf_nodes=25, random_state=42)

# GridSearchCV
param_grid = {
    'n_estimators': [96],
    'max_depth': [6],
    'min_samples_split': [100],
    'min_samples_leaf': [80],
    'bootstrap': [True],
    'max_features': ['sqrt'],
    'criterion': ['entropy'],
    'class_weight': ['balanced_subsample']
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           scoring='recall', cv=8, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

cv_scores = cross_val_score(best_rf, X_train, y_train, cv=12, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
cmd.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot a single random tree from the forest
random_tree = np.random.choice(best_rf.estimators_)
plt.figure(figsize=(20, 10))
plot_tree(random_tree, feature_names=X_train.columns, class_names=best_rf.classes_.astype(str), filled=True)
plt.title("Random Tree from the Random Forest")
plt.show()

def generate_virtual_population(tree, feature_names, n_samples=100):
    population = []
    for _ in range(n_samples):
        instance = {}
        node = 0
        while True:
            feature_index = tree.tree_.feature[node]
            if feature_index == -2:  # If it's a leaf node
                # Assign the class of this leaf
                instance['Target'] = tree.tree_.value[node].argmax()
                break
            feature_name = feature_names[feature_index]
            threshold = tree.tree_.threshold[node]

            if np.random.rand() < 0.5:
                instance[feature_name] = np.random.uniform(-np.inf, threshold)
                node = tree.tree_.children_left[node]  # Go to the left child
            else:
                instance[feature_name] = np.random.uniform(threshold, np.inf)
                node = tree.tree_.children_right[node]  # Go to the right child

        # Fill missing features with random values
        for feature in feature_names:
            if feature not in instance:
                instance[feature] = np.random.uniform(X_train[feature].min(), X_train[feature].max())

        population.append(instance)

    return pd.DataFrame(population)

virtual_population = generate_virtual_population(random_tree, X_train.columns, n_samples=100)

output_path = '../../datasets/virtual.csv'
virtual_population.to_csv(output_path, index=False)
print(f"Virtual population saved to {output_path}")
