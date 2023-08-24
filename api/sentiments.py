# Import necessary libraries
from googleapiclient.discovery import build
import pandas as pd
from decouple import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Get API key from environment variables
api_key = config("API_KEY")

# Create a YouTube API client
youtube = build("youtube", "v3", developerKey=api_key)

# Function to retrieve comments from a YouTube video
def get_youtube_video_comments(video_id: str):
    """
    This function takes a YouTube video ID and returns comments information in a pandas DataFrame.

    Args:
        video_id (str): ID of the YouTube video.

    Returns:
        pandas.DataFrame: Contains columns - 'Comment ID', 'Author', 'Comment Text', 'Date', 'Like Count'.
    """
    # Retrieve comments for the given video
    response = youtube.commentThreads().list(part="snippet,replies", maxResults=100, videoId=video_id).execute()

    comments = []

    # Extract relevant comment information from the response
    for item in response["items"]:
        comment_info = {
            "Comment ID": item["id"],
            "Author": item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
            "Comment Text": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
            "Published At": item["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
            "Like Count": item["snippet"]["topLevelComment"]["snippet"]["likeCount"],
        }

        comments.append(comment_info)

    # Create a pandas DataFrame from the comment information
    df = pd.DataFrame(comments)
    return df

# Sentiment analysis model setup
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    """
    This function calculates sentiment scores using the Roberta model.

    Args:
        example (str): Text input for sentiment analysis.

    Returns:
        dict: Sentiment scores for negative, neutral, and positive sentiments.
    """
    # Encode the input text using the tokenizer
    encoded_text = tokenizer(example, return_tensors='pt')

    # Pass the encoded text through the model
    output = model(**encoded_text)

    # Extract and normalize sentiment scores
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Create a dictionary with sentiment scores
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

print(polarity_scores_roberta("I am so happyyyy!!"))