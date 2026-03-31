
import nltk
nltk.download('punkt')

def preprocess(df):
    df["clean"]=df["Tweet"].str.lower()
    df["tokens"]=df["clean"].apply(nltk.word_tokenize)
    return df
