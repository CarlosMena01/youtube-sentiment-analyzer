from googleapiclient.discovery import build
import pandas as pd
from decouple import config

api_key = config("API_KEY")
youtube = build("youtube", "v3", developerKey=api_key)


def get_youtube_video_comments(video_id:str):
    """This function use an video Id from YouTube and return a pandas df with the comments information

    Args:
        video_id (str): Id of the YouTube video

    Returns:
        pandas.df: [ID][Author][Comment text][Date][Like counter]
    """
    response = youtube.commentThreads().list(part="snippet,replies", maxResults=100, videoId=video_id).execute()

    comments = []

    for item in response["items"]:
        comment_info = {
            "Comment ID": item["id"],
            "Author": item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
            "Comment Text": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
            "Published At": item["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
            "Like Count": item["snippet"]["topLevelComment"]["snippet"]["likeCount"],
        }

        comments.append(comment_info)

    df = pd.DataFrame(comments)
    return df
