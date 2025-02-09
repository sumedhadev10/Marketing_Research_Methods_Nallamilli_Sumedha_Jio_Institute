import praw
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Authenticate with Reddit API
reddit = praw.Reddit(
    client_id='o207oFz4cKU2UdFvYQ2qw',  # Client ID from your app
    client_secret='Y8qyet5JoiodTKSmOSmRMiWwlykVA',  # Secret from your app
    user_agent='Reddit Sentiment Analysis by /u/Successful-Owl-1041'  # Replace with your username
)

# Step 2: Choose a subreddit to analyze
subreddit_name = 'python'  # Replace with your favorite subreddit
subreddit = reddit.subreddit(subreddit_name)

# Step 3: Extract posts and comments
posts_data = []
for post in subreddit.hot(limit=50):  # Adjust the limit as needed
    post_comments = post.comments
    post_comments.replace_more(limit=0)  # Flatten nested comments

    for comment in post_comments.list():
        posts_data.append({
            'post_title': post.title,
            'post_score': post.score,
            'comment': comment.body,
            'comment_score': comment.score
        })

# Convert to a DataFrame
df = pd.DataFrame(posts_data)

# Step 4: Perform Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

df['sentiment_score'] = df['comment'].apply(analyze_sentiment)
df['sentiment'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
)

# Step 5: Visualize Sentiment Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment', order=['Positive', 'Neutral', 'Negative'])
plt.title(f'Sentiment Analysis of Comments in r/{subreddit_name}')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.show()

# Step 6: Save Data to CSV
df.to_csv(f'{subreddit_name}_sentiment_analysis.csv', index=False)
print(f"Data saved to {subreddit_name}_sentiment_analysis.csv")
