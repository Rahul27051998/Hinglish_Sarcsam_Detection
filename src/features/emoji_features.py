
import emoji
def count_emojis(text):
    return sum(1 for c in text if c in emoji.EMOJI_DATA)
def emoji_sentiment(text):
    pos=["😂","🤣","😊","🙂"]
    neg=["🙄","😒"]
    score=0
    for p in pos: score+=text.count(p)
    for n in neg: score-=text.count(n)
    return score
