

import re

# Define your API key
API_KEY = "AIzaSyD-irLwoC_-BVlvVmGVeSPpnovb49Y-Tws"

# Extract video ID from YouTube URL
def extract_video_id(url):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError("Invalid YouTube URL. Please check the URL format.")

# Initialize the YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Function to fetch all comments
def get_video_comments(video_url, max_results_per_page=100):
    # Extract video ID from URL
    video_id = extract_video_id(video_url)
    print(f"Extracted Video ID: {video_id}")
    
    comments = []
    next_page_token = None

    while True:
        # Call the API to get comments
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results_per_page,
            pageToken=next_page_token
        ).execute()

        # Extract comments
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "Author": comment['authorDisplayName'],
                "Comment": comment['textDisplay'],
                "Likes": comment['likeCount'],
                "Published At": comment['publishedAt']
            })

        # Check if there is a next page
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return pd.DataFrame(comments)

# Perform sentiment analysis
def analyze_sentiment(comments_df):
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # Returns sentiment polarity score (-1 to 1)
    
    comments_df['Sentiment Score'] = comments_df['Comment'].apply(get_sentiment)
    comments_df['Sentiment'] = comments_df['Sentiment Score'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
    )
    return comments_df

# Provide the YouTube URL
youtube_url = "https://youtu.be/U0svrc0TDQM?si=JNO5udWiLJGXET8w"  # Replace with your desired YouTube URL

# Fetch comments
comments_df = get_video_comments(youtube_url)

# Perform sentiment analysis
comments_with_sentiment = analyze_sentiment(comments_df)

# Save comments and sentiment analysis to an Excel file
excel_file = "youtube_comments_with_sentiment.xlsx"
comments_with_sentiment.to_excel(excel_file, index=False, engine='openpyxl')

# Display a preview of the extracted comments and sentiment analysis
print("First 500 rows of comments with sentiment analysis:")
print(comments_with_sentiment.head(10))

print(f"\nAll comments and sentiment analysis have been saved to '{excel_file}'.")
