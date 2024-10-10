import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

file_path = '/home/smgb/College/project/diabetes_detection/datasets/main.csv' 
data = pd.read_csv(file_path)

label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    if col != 'Target':
        data[col] = label_encoder.fit_transform(data[col])

X = data.drop('Target', axis=1)  
y = data['Target']               

cart_model = DecisionTreeClassifier(random_state=42, criterion='gini')  #CART (Gini)
dt_model = DecisionTreeClassifier(random_state=42, criterion='entropy')  #Decision Tree (Entropy)

voting_ensemble = VotingClassifier(estimators=[
    ('cart', cart_model),
    ('dt', dt_model)
], voting='hard')

bagging_cart = BaggingClassifier(estimator=cart_model, n_estimators=150, random_state=23052005)

kf = KFold(n_splits=15, shuffle=True, random_state=23052005)

voting_cv_scores = cross_val_score(voting_ensemble, X, y, cv=kf, scoring='accuracy')
print("Voting Classifier Cross-Validation Accuracy (15-Fold):", voting_cv_scores.mean())

bagging_cv_scores = cross_val_score(bagging_cart, X, y, cv=kf, scoring='accuracy')
print("Bagging CART Cross-Validation Accuracy (15-Fold):", bagging_cv_scores.mean())

voting_ensemble.fit(X, y)  
bagging_cart.fit(X, y)  

y_pred_voting = voting_ensemble.predict(X)
y_pred_bagging = bagging_cart.predict(X)

accuracy_voting = accuracy_score(y, y_pred_voting)
accuracy_bagging = accuracy_score(y, y_pred_bagging)

report_voting = classification_report(y, y_pred_voting)
report_bagging = classification_report(y, y_pred_bagging)

print("Final Voting Classifier Accuracy:", accuracy_voting)
print("Final Bagging Classifier Accuracy:", accuracy_bagging)
print("Voting Classifier Classification Report:\n", report_voting)
print("Bagging Classifier Classification Report:\n", report_bagging)