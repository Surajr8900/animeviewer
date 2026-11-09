
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "anime.csv")

df = pd.read_csv(CSV_PATH)

# Validate columns
required = ["anime_id", "name", "genre"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Clean dataset
df = df[required].copy()
df = df.dropna(subset=["name"])
df["genre"] = df["genre"].fillna("")

# Create tags
df["tags"] = df["genre"].astype(str).str.lower()

new_df = df[["anime_id", "name", "tags"]].copy()
new_df = new_df.reset_index(drop=True)

# Vectorization
cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(new_df["tags"])

# Similarity
similarity = cosine_similarity(vectors)

# Save model files
with open(os.path.join(BASE_DIR, "anime_dict.pkl"), "wb") as f:
    pickle.dump(new_df.to_dict(), f)

with open(os.path.join(BASE_DIR, "anime_similarity.pkl"), "wb") as f:
    pickle.dump(similarity, f)

print("Model trained successfully!")
print("Total anime:", len(new_df))
print("Similarity shape:", similarity.shape)