import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

# Read the CSV file into a DataFrame
data = pd.read_csv("salaries.csv")

# Split into inputs and target variables
inputs = data.drop('salary_more_than_100k', axis='columns')
target = data['salary_more_than_100k']

# Perform label encoding for categorical variables
label_encoder = LabelEncoder()
inputs['company_n'] = label_encoder.fit_transform(inputs['company'])
inputs['job_n'] = label_encoder.fit_transform(inputs['job'])
inputs['degree_n'] = label_encoder.fit_transform(inputs['degree'])

# Drop the original categorical columns
inputs = inputs.drop(['company','job','degree'], axis='columns')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

# Create the decision tree classifier model
model = tree.DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Evaluate the model's accuracy on the training set
train_accuracy = model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

# Evaluate the model's accuracy on the testing set
test_accuracy = model.score(X_test, y_test)
print("Testing Accuracy:", test_accuracy)

# Make predictions
prediction1 = model.predict([[2, 1, 0]])
prediction2 = model.predict([[2, 1, 1]])
print("Prediction 1:", prediction1)
print("Prediction 2:", prediction2)
