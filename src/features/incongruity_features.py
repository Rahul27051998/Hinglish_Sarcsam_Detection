
from textblob import TextBlob
def sarcasm_incongruity(text):
    polarity=TextBlob(text).sentiment.polarity
    if "😂" in text and polarity<0: return 1
    if "🙄" in text and polarity>0: return 1
    return 0
