import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

'''
pandas: Used for handling and manipulating datasets. It reads the CSV file and organizes the data in a DataFrame format.
train_test_split: This function from sklearn splits your dataset into training and testing subsets. The model is trained on one part and tested on another, ensuring that the model generalizes well to new data.
LogisticRegression: This is the logistic regression model from sklearn. It’s a linear model used for binary or multi-class classification tasks.
StandardScaler: A scaling technique that standardizes features by removing the mean and scaling to unit variance. It ensures features are on a similar scale, which improves model performance for algorithms that are sensitive to feature scaling (like logistic regression).
accuracy_score: This metric from sklearn evaluates the model by calculating the proportion of correctly classified examples in the test set.
classification_report: This function generates a detailed report showing the main classification metrics such as precision, recall, and F1-score for each class in the dataset.
'''


def perform_logistic_regression(input_file):
    df = pd.read_csv(input_file)

    # Separate features (X) and the target (y)
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''
    train_test_split(): This function splits the data into two sets:
    Training set (X_train, y_train): Used to train the logistic regression model.   
    Test set (X_test, y_test): Used to evaluate how well the model performs on unseen data.
    test_size=0.2: This specifies that 20% of the data will be used for testing, while 80% will be used for training.
    random_state=42: This sets a seed for reproducibility, so that the split will be the same every time you run the program.
    '''

    # Standardize the features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

# Usage example
input_file = '../datasets/data.csv'  # Replace with your file path
perform_logistic_regression(input_file)

