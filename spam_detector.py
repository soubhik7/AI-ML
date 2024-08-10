# Import necessary libraries
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Loading Dataset
#url = 'https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv'
#response = requests.get(url)
#data = pd.read_csv(StringIO(response.text), encoding='latin-1')

file_path = '/Users/soubhik/Desktop/spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')



print(data.head())

# Clean columns by dropping unnecessary ones
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.columns = ['label', 'text']

# Display the cleaned DataFrame
print(data.head())

# Exploratory Data Analysis (EDA)
# Check for missing values
print(data.isna().sum())

# Check the shape of the data
print(data.shape)

# Check the balance of the target labels
data['label'].value_counts(normalize=True).plot.bar()

# Train-test-split
# Create Feature (X) and Label (y) sets
X = data['text']
y = data['label']

# Split the data into training (66%) and testing (33%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

# Display the shape of the training and testing sets
print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)

# Feature Extraction (Bag of Words Model)
# Create a Bag of Words model using CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# Display the shape of the transformed training data
print(X_train_cv.shape)

# Model Training
# Train a Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train_cv, y_train)

# Model Prediction
# Transform the testing data using the fitted CountVectorizer
X_test_cv = cv.transform(X_test)

# Create a prediction set
predictions = lr.predict(X_test_cv)

# Display the predictions
print(predictions)

# Model Evaluation
# Create a confusion matrix
df = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['ham', 'spam'], columns=['ham', 'spam'])

# Display the confusion matrix
print(df)