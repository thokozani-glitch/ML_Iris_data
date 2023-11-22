import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import json
import joblib

# Load the Iris dataset
data = pd.read_csv('Iris.csv')

# Drop unnecessary columns
data = data.drop(['Id'], axis=1)

# Extract the target column 'Species'
target = data['Species']

# Drop the target column for scaling
data = data.drop(['Species'], axis=1)

# Standardize the features using StandardScaler
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Add the 'Species' column back to the scaled data
data_scaled['Species'] = target

# Split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(
    data_scaled.drop(['Species'], axis=1), 
    data_scaled['Species'], 
    test_size=0.2, 
    random_state=42
)

# Initialize the Support Vector Machine (SVM) classifier with a linear kernel
svm_classifier = SVC(kernel='linear', random_state=42)

# Initialize RFECV with the linear SVM classifier and cross-validated scoring
rfecv = RFECV(estimator=svm_classifier, step=1, cv=5, scoring='accuracy')

# Fit RFECV to the training data
rfecv.fit(train_data, train_target)

# Get the selected features
selected_features = train_data.columns[rfecv.support_]

# Transform the training and testing data using the selected features
train_data_selected = rfecv.transform(train_data)
test_data_selected = rfecv.transform(test_data)

# Train the classifier on the selected features
svm_classifier.fit(train_data_selected, train_target)

# Make predictions on the testing data
predictions = svm_classifier.predict(test_data_selected)

# Evaluate the performance of the model
accuracy = accuracy_score(test_target, predictions)
conf_matrix = confusion_matrix(test_target, predictions)
classification_rep = classification_report(test_target, predictions)

# Save the selected features as a JSON file
selected_features_json = json.dumps(list(selected_features))
with open('selected_features.json', 'w') as json_file:
    json_file.write(selected_features_json)

# Save the trained SVM classifier with selected features using joblib
joblib.dump(svm_classifier, 'svm_model_with_selected_features.joblib')

# Streamlit web app
st.title('SVM Model Web App')

# Display the selected features
st.subheader('Selected Features:')
st.write(selected_features)

# Display evaluation metrics
st.subheader('Model Evaluation Metrics:')
st.write(f"Model Accuracy with Selected Features: {accuracy}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
st.write("Classification Report:")
st.write(classification_rep)
