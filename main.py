import pandas as pd
import joblib
from tqdm import tqdm

from src.preprocessing import preprocess
from src.features.feature_engineering import build_features
from src.training.train_models import train_models
from src.evaluation.evaluate import evaluate


print("\n==============================")
print("Loading datasets...")
print("==============================")

train_df = pd.read_csv("data/raw/unique tweets_train.csv")
test_df = pd.read_csv("data/raw/unique tweets_test.csv")


print("\n==============================")
print("Preprocessing datasets...")
print("==============================")

train_df = preprocess(train_df)
test_df = preprocess(test_df)


print("\n==============================")
print("Building Features...")
print("==============================")

X_train, y_train, vec = build_features(train_df)
X_test = vec.transform(test_df["clean"])
y_test = test_df["Label"]


# ===============================
# NEW FEATURE ENGINEERING STATUS
# ===============================

print("\n==============================")
print("Feature Engineering Completed ✔")
print("==============================")

print("Emoji Features Generated ✔")
print("Saved at: results/emoji_features_train.csv")

print("Incongruity Features Generated ✔")
print("Saved at: results/incongruity_features_train.csv")


print("\n==============================")
print("Training 10 Models (20 Epochs Each)...")
print("==============================\n")

# Progress visualization handled inside train_models (tqdm per model)
models, best_model, best_name = train_models(X_train, y_train, X_test, y_test)


print("\n==============================")
print("Accuracy Graphs Generated Successfully ✔")
print("==============================")


print("\n==============================")
print("Evaluating Models on Test Data...")
print("==============================\n")

# Run evaluation (this saves CSV + confusion matrices)
evaluate(models, X_test, y_test)


# ===============================
# Display Evaluation Table
# ===============================

print("\n==============================")
print("Evaluation Metrics (Test Data)")
print("==============================\n")

metrics_df = pd.read_csv("results/test_metrics.csv")

print(metrics_df.to_string(index=False))


print("\n==============================")
print("Evaluation Metrics Saved ✔")
print("Location: results/test_metrics.csv")
print("==============================")


print("\n==============================")
print("Confusion Matrices Generated ✔")
print("Location: results/confusion_matrices/")
print("==============================")


print("\n==============================")
print("Saving Best Model...")
print("==============================")

joblib.dump(best_model, "src/models/best_model.pkl")
joblib.dump(vec, "src/models/vectorizer.pkl")


print(f"\nBest Model Selected: {best_name}")


print("\n==============================")
print("Pipeline Execution Completed Successfully ✔")
print("==============================\n")