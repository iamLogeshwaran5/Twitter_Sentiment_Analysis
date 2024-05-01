# Twitter Sentiment Analysis

## Introduction
This project analyzes the sentiments expressed in tweets from the social networking site Twitter. Using a dataset of 1.6 million tweets, scraped using the Twitter API and labeled for sentiment, the project's goal is to build a classification model that can accurately predict the sentiment of a tweet as negative, neutral, or positive.

## Dataset Description
The dataset contains 1,600,000 tweets, each annotated with sentiment labels, and includes the following fields:
- **target**: Polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- **ids**: Unique identifier for the tweet
- **date**: Timestamp of the tweet (e.g., Sat May 16 23:58:44 UTC 2009)
- **flag**: Query flag (NO_QUERY if there is no associated query)
- **user**: Username of the tweeter
- **text**: Text content of the tweet

## Data Loading and Preprocessing
The data was loaded using the Pandas library, and several preprocessing steps were applied to prepare the data for analysis, including:
- Removing special characters and URLs
- Converting all text to lowercase
- Tokenizing text and removing stop words
- Applying lemmatization


## Exploratory Data Analysis (EDA)
Exploratory analysis was performed to understand the distribution of sentiment classes and identify any biases or trends in the dataset. Visualization of sentiment distribution showed a balanced dataset across negative, neutral and positive sentiments.

## Feature Engineering 
Features were engineered using TF-IDF vectorization to transform text data into a format suitable for model training. This method captures the importance of words relative to their frequency across all documents.


