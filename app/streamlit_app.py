
import streamlit as st
import joblib
import sys
import os

# Add project root to path (important for imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.emoji_features import count_emojis, emoji_sentiment
from src.features.incongruity_features import sarcasm_incongruity

# Load best model and vectorizer
model = joblib.load("src/models/best_model.pkl")
vec = joblib.load("src/models/vectorizer.pkl")

st.title("Advanced Hinglish Sarcasm Detection")

st.write("This system uses:")
st.write("• Best ML model")
st.write("• Emoji features")
st.write("• Incongruity detection")

text = st.text_area("Enter Tweet")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter a tweet.")
    else:

        # =========================
        # ML Prediction
        # =========================
        X = vec.transform([text])
        pred = model.predict(X)[0]

        # =========================
        # Feature Extraction
        # =========================
        emoji_count_val = count_emojis(text)
        emoji_score = emoji_sentiment(text)
        incongruity = sarcasm_incongruity(text)

        # =========================
        # Convert model prediction
        # =========================
        model_signal = str(pred).lower() in ["1", "yes", "sarcastic"]

        # =========================
        # Feature-based signals
        # =========================
        emoji_signal = emoji_score != 0
        incongruity_signal = incongruity == 1

        # =========================
        # Combined Decision Logic
        # =========================
        sarcasm_score = sum([
            model_signal,
            emoji_signal,
            incongruity_signal
        ])

        # Final decision
        if sarcasm_score >= 2:
            st.error("Sarcastic Tweet 😏")
        else:
            st.success("Non-Sarcastic Tweet 🙂")

        # =========================
        # Display Feature Insights
        # =========================
        st.subheader("Analysis Details")

        st.write("Model Prediction:", pred)
        st.write("Emoji Count:", emoji_count_val)
        st.write("Emoji Sentiment Score:", emoji_score)
        st.write("Incongruity Feature:", incongruity)
        st.write("Sarcasm Signals Detected:", sarcasm_score, "/ 3")