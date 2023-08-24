from googleapiclient.discovery import build

import pandas as pd

from decouple import config

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

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

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

print(polarity_scores_roberta("I am so happyyyy!!"))