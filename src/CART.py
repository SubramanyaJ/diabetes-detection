#CART
#Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


#Loading the dataset
data = pd.read_csv('main.csv')

#Converting the targets to numerical values
label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

#Splitting the given data as training and testing scenarios
X = data.drop(columns=['Target'])
y = data['Target']


#Splitting the dataset: 40% for training, 60% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5355, random_state=69)


#Training the decision tree classifier
tree_clf = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=296,
    class_weight={0:3, 1:1, 2:1, 3:3, 4:2, 5:6, 6:7},
    ccp_alpha=0.00000375,
    random_state=69)


tree_clf.fit(X_train, y_train)

#Evaluating the model with these stats
y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

#Printing the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
