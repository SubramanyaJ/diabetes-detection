import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

file_path = '/home/smgb/College/project/diabetes_detection/datasets/main.csv' 
data = pd.read_csv(file_path)

print(data.info())
print(data.head())  

label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    if col != 'Target':  
        data[col] = label_encoder.fit_transform(data[col])

plt.figure(figsize=(12, 8))
corr_matrix = data.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

threshold = 0.000000234
corr_with_target = corr_matrix['Target'].abs()
selected_features = corr_with_target[corr_with_target > threshold].index.drop('Target')

print("Selected Features based on correlation threshold:", selected_features)

X = data[selected_features]  
y = data['Target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6254, random_state=2713)


rf_model = RandomForestClassifier(
    max_depth=19,
    min_samples_split=266,
    ccp_alpha=0.00000375,
    random_state=2713
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Random Forest Model Accuracy:", accuracy)
print("Classification Report:\n", report)

