import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Build the CART model with adjusted hyperparameters
cart_model = DecisionTreeClassifier(
    max_depth= None,              # Limit the depth of the tree
    min_samples_split=2,      # Minimum samples to split a node
    min_samples_leaf=1,        # Minimum samples required in a leaf
    criterion='gini',          # Use Gini impurity to make splits
    random_state= None,            # Ensure reproducibility
    splitter='best',
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None

)

# Train the model
cart_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = cart_model.predict(X_test_scaled)

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

