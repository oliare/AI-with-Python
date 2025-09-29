import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import json
import re

nltk.download("stopwords")
nltk.download("wordnet")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

df = pd.read_csv("../assets/product_reviews_list.csv", encoding="utf-8-sig")

processed_comments = []
sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

for index, row in df.iterrows():
    text = str(row["Review Text"])

    tokens = [w.lower() for w in re.findall(r'\b[a-z]+\b', text.lower())]

    filtered = [word for word in tokens if word not in stop_words]

    stemmed = [stemmer.stem(word) for word in filtered]
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]

    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    sentiment_counts[sentiment] += 1

    processed_comments.append({
        "original": text,
        "tokens": tokens,
        "filtered": filtered,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
        "sentiment": sentiment
    })

with open("processed_comments.json", "w", encoding="utf-8") as f:
    json.dump(processed_comments, f, ensure_ascii=False, indent=4)

pd.DataFrame(processed_comments).to_csv("processed_comments.csv", index=False, encoding="utf-8-sig")

print("Done! Reviews processed:", len(processed_comments))
print("Sentiment analysis results:", sentiment_counts)
