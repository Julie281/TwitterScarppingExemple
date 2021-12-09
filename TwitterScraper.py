import tweepy 
import pandas as pd 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
sw = set(stopwords.words('french'))
sw.add('RT')
sw.add('a')
sw.add('au')
sw.add('dont')
import numpy as np
import re  


consumer_key = "xx"
consumer_secret = "xx"
access_token = "xx"
access_token_secret = "xx"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

text_query = 'presidentielle'
count = 150

# Creation of query method using parameters
tweets = tweepy.Cursor(api.search_tweets,q=text_query).items(count)
 
 # Pulling information from tweets iterable object
tweets_list = [ [tweet.text] for tweet in tweets]
 
 # Creation of dataframe from tweets list
 # Add or remove columns as you remove tweet information
tweets_df = pd.DataFrame(tweets_list)

#print(tweets_df)

all_sentences = []

for word in tweets_df:
    all_sentences.append(word)

all_sentences
df1 = tweets_df.to_string()

tweets_df_split = df1.split()

#print(tweets_df_split)

tweets_df_split = [re.sub(r'[^A-Za-z0-9áàâäãåçéèêëíìîïñóòôöõúùûüýÿæœÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝŸÆŒ]+', '', x) for x in tweets_df_split]
#print(tweets_df_split)
tweets_df_split

tweets_df_split2 = []

for word in tweets_df_split:
    if word != '':
        tweets_df_split2.append(word)


stem = tweets_df_split2

stem2 = []

for word in stem:
    if word not in sw:
        stem2.append(word)

print(stem2)

df = pd.DataFrame(stem2)

df = df[0].value_counts()

from nltk.probability import FreqDist

freqdoctor = FreqDist()

for words in df:
    freqdoctor[words] += 1

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = df[:20,]
plt.figure(figsize=(10,5))
sns.barplot(df.values, df.index, alpha=0.8)
plt.title('Top Words Overall')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()
