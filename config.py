# --- Configuration File ---

# Target user for recommendations
[cite_start]TARGET_USER_ID = 40 [cite: 140, 249]

# --- Similarity Step Config ---
# Minimum common movies rated by two users to be considered similar
[cite_start]MIN_COMMON_USERS = 3 [cite: 141]

# --- Prediction Step Config ---
[cite_start]NUM_RECOMMENDATIONS = 5 [cite: 250]
[cite_start]MIN_CONTRIBUTORS = 5 [cite: 251]
[cite_start]MIN_RATING = 0.5 [cite: 252]
[cite_start]MAX_RATING = 5.0 [cite: 253]

# --- Data Paths ---
[cite_start]DATASET_REPO = "https://github.com/sankalpjain99/Movie-recommendation-system.git"
[cite_start]RATINGS_CSV_PATH = "Movie-recommendation-system/ratings.csv"
[cite_start]MOVIES_CSV_PATH = "Movie-recommendation-system/movies.csv"
