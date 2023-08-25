import re
import pandas as pd
from sentiments import get_youtube_video_comments, polarity_scores_roberta

def extract_video_id_from_url(url:str):
    """
    Extracts the video ID from a YouTube video URL.

    Args:
        url (str): The YouTube video URL.

    Returns:
        str or None: The extracted video ID or None if not found.
    """
    # Regular expression to match the video ID in a YouTube URL
    pattern = r"(?<=v=)[a-zA-Z0-9_-]+"
    
    # Use re.search to find the video ID in the URL
    match = re.search(pattern, url)
    
    if match:
        # Extract and return the video ID
        video_id = match.group()
        return video_id
    else:
        return None

def get_sentiment_scores_for_video(video_url:str):
    """
    Gets sentiment scores for comments on a YouTube video on a DF.

    Args:
        video_url (str): The URL of the YouTube video.

    Returns:
        pandas.DataFrame or str: A DataFrame containing comments and sentiment scores,
        or an error message if the URL is invalid.
    """
    video_id = extract_video_id_from_url(video_url)    
    
    if not video_id:
        return "Error: Invalid URL"
    
    df_comments = get_youtube_video_comments(video_id)

    # Create a DataFrame with the sentiment scores
    df_sentiments = df_comments.apply(polarity_scores_roberta, axis=1).apply(pd.Series)

    df_results = pd.concat([df_comments, df_sentiments], axis=1)

    return df_results

