
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.emoji_features import count_emojis, emoji_sentiment
from src.features.incongruity_features import sarcasm_incongruity

def build_features(df):
    vec=TfidfVectorizer(max_features=3000)
    X=vec.fit_transform(df["clean"])
    y=df["Label"]

    pd.DataFrame({
        "emoji_count":df["clean"].apply(count_emojis),
        "emoji_sentiment":df["clean"].apply(emoji_sentiment)
    }).to_csv("results/emoji_features_train.csv",index=False)

    pd.DataFrame({
        "incongruity":df["clean"].apply(sarcasm_incongruity)
    }).to_csv("results/incongruity_features_train.csv",index=False)

    return X,y,vec
