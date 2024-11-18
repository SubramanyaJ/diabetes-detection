# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv('/home/smgb/College/project/diabetes_detection/src/data.csv')

# Ignoring type 4 diabetes entries for the general classification model
filtered_data = data[data['Target'] != 4].reset_index(drop=True)

# Defining features and target variable for general diabetes types
X = filtered_data.drop(columns=['Target'])
y = filtered_data['Target']

# Splitting the dataset: 53.55% for training, 46.45% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5355, random_state=69)

# Training the Random Forest classifier for general diabetes types (excluding type 4)
forest_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=296,
    class_weight={0: 3, 1: 1, 2: 1, 3: 3, 4: 2, 5: 6, 6: 7},
    random_state=69
)
forest_clf.fit(X_train, y_train)

# Predicting on the test data for general diabetes types
y_pred = forest_clf.predict(X_test)

# Accuracy for general diabetes types
accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy for General Diabetes Types (Excluding type 4): {accuracy * 100:.2f}%\n")

# Generating and printing classification report for general diabetes types
report = classification_report(y_test, y_pred, zero_division=0)
# print("Classification Report for General Diabetes Types:\n", report)

# --- Generate and Save Feature Importance Plot ---
# feature_importances = pd.Series(forest_clf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plotting feature importances
# plt.figure(figsize=(12, 8))
# sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
# plt.title("Feature Importances for General Diabetes Types")
# plt.xlabel("Importance Score")
# plt.ylabel("Feature")
# plt.tight_layout()

# Saving the feature importance plot
# feature_importance_path = 'feature_importances(RNDTR).png'
# plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Feature importance plot saved at: {feature_importance_path}")

# --- Section for type 4 Diabetes Classification ---
# Filter the data to include only type 4 diabetes entries
type_4_data = data[data['Target'] == 4]

# Defining features and target variable for type 4 diabetes
X_type4 = type_4_data.drop(columns=['Target'])
y_type4 = type_4_data['Target']

# Splitting the dataset for type 4 diabetes: 70% training, 30% testing
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_type4, y_type4, train_size=0.5355, random_state=42)

# Training a Random Forest classifier for type 4 diabetes
type4_forest_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    random_state=42
)
type4_forest_clf.fit(X_train_4, y_train_4)

# Predicting on the test data for type 4 diabetes
y_pred_4 = type4_forest_clf.predict(X_test_4)

# Accuracy for type 4 diabetes
accuracy_type4 = accuracy_score(y_test_4, y_pred_4)
# print(f"Accuracy for Type 4 Diabetes: {accuracy_type4 * 100:.2f}%\n")

# Generating and printing classification report for type 4 diabetes
report_type4 = classification_report(y_test_4, y_pred_4, zero_division=0)
# print("Classification Report for Type 4 Diabetes:\n", report_type4)

# After generating individual classification reports for general and type 4 diabetes:
# print("Classification Report for General Diabetes Types:\n", report)
# print("Classification Report for Type 4 Diabetes:\n", report_type4)

# --- Unified Classification and Metrics Output ---
# Paste the code provided here
combined_y_true = pd.concat([y_test, y_test_4]).reset_index(drop=True)
combined_y_pred = pd.concat([pd.Series(y_pred), pd.Series(y_pred_4)]).reset_index(drop=True)

# Calculate overall accuracy
combined_accuracy = accuracy_score(combined_y_true, combined_y_pred)
print(f"Overall Accuracy (All Types): {combined_accuracy * 100:.2f}%\n")

# Generate the combined classification report
combined_report = classification_report(
    combined_y_true, combined_y_pred, zero_division=0,
    target_names=[f"Type {i}" for i in sorted(combined_y_true.unique())]
)
print("Combined Classification Report (All Types):\n", combined_report)

# Generate the combined confusion matrix
conf_matrix = confusion_matrix(combined_y_true, combined_y_pred)

# Plot the combined confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[f"Type {i}" for i in sorted(combined_y_true.unique())],
            yticklabels=[f"Type {i}" for i in sorted(combined_y_true.unique())])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (All Types)")

# Save the confusion matrix plot
conf_matrix_path = 'confusion_matrix_combined.png'
plt.savefig(conf_matrix_path)
plt.close()

print(f"Confusion matrix image saved at: {conf_matrix_path}")