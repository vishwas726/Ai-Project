# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("sentimentalreview_IBMpro.tsv", delimiter='\t', quoting=3)

# Data Exploration
print(df.head())
print(df.info())
print(df['Liked'].value_counts())

# Text Preprocessing
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

def text_process(msg):
    nopunc = [char for char in msg if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

df['tokenized_Review'] = df['Review'].apply(text_process)

# Creating a Word Cloud
from wordcloud import WordCloud

word_cloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['Liked'] == 1]['tokenized_Review']))
plt.figure(figsize=(10, 5))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Reviews")
plt.show()

word_cloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['Liked'] == 0]['tokenized_Review']))
plt.figure(figsize=(10, 5))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Reviews")
plt.show()

# Text Vectorization
vectorizer = CountVectorizer(max_df=0.9, min_df=10)
X = vectorizer.fit_transform(df['tokenized_Review']).toarray()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, df['Liked'], test_size=0.2, random_state=107)

# XGBoost Model
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

# Predictions
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Evaluate the model
print("\nTraining Classification Report:")
print(classification_report(y_train, y_pred_train))

print("\nTesting Classification Report:")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
print("\nConfusion Matrix for Testing Data:")
print(confusion_matrix(y_test, y_pred_test))

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
