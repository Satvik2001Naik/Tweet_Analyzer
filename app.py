import tweepy as tw
import numpy as np
import pandas as pd
import io
import re
import emoji
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = stopwords.words('english')

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from flask import Flask, render_template, request, Response

app = Flask(__name__)

key = 'K6gd1YLuinBhwza3hsd4J0q8x'
secret = '64CLjWg5WlAz2FobRCNDUNhwLly06DeN5reTeI7qG64xtYrbYD'
access_token = '1505776500527988738-PLscmTb4lsnPlb18B84CHemOB3xoYl'
access_token_secret = 'SU7yC9BtKgrivpoCjrmms2zYyI2NBMya02zPKbRotnZNa'

# Authentication
auth = tw.OAuthHandler(key,secret)
auth.set_access_token(access_token,access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

def clean_tweets(tweet):
    rm_rt = re.sub('RT\s+'," ",tweet)
    rm_at = re.sub('\B@\w+'," ",rm_rt)
    rm_hash = re.sub('\B#\w+'," ",rm_at)
    rm_emo = emoji.demojize(rm_hash)
    return rm_emo

def preprocess_tweets(tweet, custom_stopwords):
    preprocessed_tweet = tweet
    preprocessed_tweet.replace('[^\w\s]','')
    preprocessed_tweet = " ".join(word for word in preprocessed_tweet.split() if word not in stop_words)
    preprocessed_tweet = " ".join(word for word in preprocessed_tweet.split() if word not in custom_stopwords)
    preprocessed_tweet = " ".join(Word(word).lemmatize() for word in preprocessed_tweet.split())
    return(preprocessed_tweet)

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

def sentiment_analyzer(hashtag, limit):
    query = tw.Cursor(api.search_tweets, q=hashtag).items(limit)
    tweets = [{'Tweets':tweet.text, 'Timestamp':tweet.created_at} for tweet in query]
    df = pd.DataFrame.from_dict(tweets)
    custom_stopwords = ['RT', hashtag]
    cleaned_tweets = df['Tweets'].apply(lambda x:clean_tweets(x))
    df['Cleaned/Preprocessed Tweet'] = cleaned_tweets.apply(lambda x:preprocess_tweets(x,custom_stopwords))
    df['polarity'] = df['Cleaned/Preprocessed Tweet'].apply(lambda x:TextBlob(x).sentiment[0])
    df['subjectivity'] = df['Cleaned/Preprocessed Tweet'].apply(lambda x:TextBlob(x).sentiment[1])
    df['sentiment'] = df['polarity'].apply(getAnalysis)
    return df

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/result', methods=["POST", "GET"])
def result():
    if request.method == "POST":
        hashtag=request.form['hashtag']
        limit=request.form['limit']
        data = sentiment_analyzer(hashtag,int(limit))
        return render_template('index.html', title = "Sentiment Results", sentiment = data['sentiment'].value_counts(0), hashtag="Hashtag:- "+request.form['hashtag'], limit="No. of records:- "+request.form['limit'], tables=[data.to_html()],titles=[''])

if __name__ == "__main__":
    app.run(debug=True)