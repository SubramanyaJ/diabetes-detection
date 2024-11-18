#Bayes Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


file_path = 'main.csv' 
data = pd.read_csv(file_path)


X = data.drop('Target', axis=1)  
y = data['Target']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4625, random_state=314)

nb_model = GaussianNB(
    var_smoothing=1e-10,
    #priors=[0.125,0.1,0.1,0.125,0.125,0.125,0.3]
)
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
nb_report = classification_report(y_test, nb_pred)

print("Naive Bayes Model Accuracy:", nb_acc)
print("Naive Bayes Model Classification Report:\n", nb_report)
