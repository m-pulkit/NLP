import re
import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.metrics import confusion_matrix, classification_report

def preprocess_tweet(tweet):
    """
    Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    
    if 'C:/Users/pulki/OneDrive/Documents/Jupyter/NLP - Deeplearning.ai/nltk_data' not in nltk.data.path:
        # add path from our local workspace containing pre-downloaded corpora files to nltk's data path
        nltk.data.path.append('C:/Users/pulki/OneDrive/Documents/Jupyter/NLP - Deeplearning.ai/nltk_data')
    
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    punctuation = string.punctuation
    
    tweet = re.sub(r'\$\w*', '', tweet)                    # remove stock market tickers like $GE
    tweet = re.sub(r'^RT[\s]+', '', tweet)                 # remove old style retweet text "RT"
    tweet = re.sub(r'@[\w]*?[\s]', '', tweet)              # remove usernames
    # remove hyperlinks
    tweet = re.sub(r'(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?./\/\.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#*', '', tweet)                       # remove hashtags
    
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet.lower())
    
    clean_tokens = [x for x in tokens if (x not in stopwords_english) & (x not in punctuation)]
    
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in clean_tokens]
    
    
    
def build_freq(tweet_list, labels):
    """
    Build frequencies.
    Input:
        tweets: a list of tweets
        labels: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dataframe mapping each word to its positive and negative sentiment counts
        frequency
    """
    wl_pair = []
    for tweet, label in zip(tweet_list, labels):
        for word in tweet:
            wl_pair.append([word, label])
    
    freq = pd.DataFrame(wl_pair, columns=['vocab', 'label'])
    
    word_freq = pd.DataFrame(freq.groupby('vocab').sum())
    
    freq.label = 1-freq.label
    word_freq['neg'] = freq.groupby('vocab').sum()
    
    return word_freq.convert_dtypes().rename(columns={'label': 'pos'}).reset_index()#.to_dict(orient='list')




def extract_features(tokenized_tweet, df):
    '''
    Input: 
        tweet: a list of words for one tweet
        df: a dataframe mapping each word to its positive and negative sentiment counts
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    return [1] + list(df.loc[df.vocab.isin(tokenized_tweet), ['pos', 'neg']].sum())



def predict_tweet(tweet, model, df):
    '''
    Input: 
        tweet: a string
        model: trained model
        df: a dataframe mapping each word to its positive and negative sentiment counts
    Output: the probability of a tweet being positive or negative
    '''
    return model.predict([extract_features(preprocess_tweet(tweet), df)])
    
    
    
def test_model(test_x, test_y, model, df):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        model: Trained Model
        df: a dataframe mapping each word to its positive and negative sentiment counts
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    y_pred = model.predict([extract_features(preprocess_tweet(tweet), df) for tweet in test_x])
    
    print('Confusion Matrix: \n', confusion_matrix(test_y, y_pred))
    print(classification_report(test_y, y_pred))
    
    return sum([x==y for x,y in zip(y_pred, test_y)]) / len(test_y)



