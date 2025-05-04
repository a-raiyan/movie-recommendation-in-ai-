# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Loading dataset
data = pd.read_csv('dataset.csv')

# Feature selection: Using 'genre' and 'rating' as features
X = data[['genre', 'rating']]

# Encoding categorical 'genre' feature using one-hot encoding
X = pd.get_dummies(X, columns=['genre'], drop_first=True)

# Target variable: movie_id
y = data['movie_id']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Making predictions and evaluating the Random Forest model
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%')

# Support Vector Machine model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Making predictions and evaluating the SVM model
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f'SVM Model Accuracy: {svm_accuracy * 100:.2f}%')
