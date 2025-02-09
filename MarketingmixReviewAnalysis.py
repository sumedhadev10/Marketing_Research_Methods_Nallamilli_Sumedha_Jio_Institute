# Importing necessary libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Loading the dataset
data = pd.read_csv('AmazonReview.csv')

# Displaying the first few rows of the dataset
print(data.head())

# Checking the data information
data.info()

# Dropping null values
data.dropna(inplace=True)

# Categorizing Sentiments
# 1, 2, 3 -> Negative (0)
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0

# 4, 5 -> Positive (1)
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1

# Cleaning the reviews
stp_words = stopwords.words('english')

def clean_review(review):
    cleanreview = " ".join(word for word in review.split() if word not in stp_words)
    return cleanreview

data['Review'] = data['Review'].apply(clean_review)

# Displaying the cleaned data
print(data.head())

# Analyzing the dataset
print(data['Sentiment'].value_counts())

# WordCloud for negative sentiments
negative_reviews = ' '.join(word for word in data['Review'][data['Sentiment'] == 0].astype(str))
wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(negative_reviews), interpolation='bilinear')
plt.axis('off')
plt.show()

# WordCloud for positive sentiments
positive_reviews = ' '.join(word for word in data['Review'][data['Sentiment'] == 1].astype(str))
wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(positive_reviews), interpolation='bilinear')
plt.axis('off')
plt.show()

# Converting text into vectors using TF-IDF
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review']).toarray()

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, data['Sentiment'], test_size=0.25, random_state=42)

# Training the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()

# Fitting the model
model.fit(x_train, y_train)

# Testing the model
pred = model.predict(x_test)

# Model accuracy
accuracy = accuracy_score(y_test, pred)
print(f"Model Accuracy: {accuracy:.5f}")

# Confusion matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])

cm_display.plot()
plt.show()
