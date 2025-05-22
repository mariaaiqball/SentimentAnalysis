from tweepy import Paginator
import tweepy
import pandas as pd 


# variables to initialize
BEARER_TOKEN = ""
'''CHANGE FILE NAME '''
FILENAME = 'sb.csv' 

# Initialize the client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

all_tweets = []

"""CHANGE QUERY"""
for response in Paginator(client.search_recent_tweets,
                          query="starbucks -is:retweet -has:links -has:mentions -is:reply lang:en", 
                          max_results=100, 
                          tweet_fields=["text"],
                          limit=6):  # 5 * 100 = 500 tweets
    for tweet in response.data:
        all_tweets.append(tweet.text)

# Create a DataFrame
df = pd.DataFrame(all_tweets, columns=["text"])
df.to_csv(FILENAME, index=False)







