import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Netflix dataset
df = pd.read_csv('Dataset.csv')

# ==========================================
# Step 1: Prepare content-related features
# ==========================================
# We select relevant text-based features and clean them (lowercase, remove commas) 
# so they can be processed effectively by the NLP model.
features = ['type', 'director', 'country', 'rating', 'listed_in']

for feature in features:
    # Convert to string, lowercase, and remove commas to treat compound genres/names cleanly
    df[feature + '_clean'] = df[feature].astype(str).str.lower().str.replace(', ', ' ')

# Combine the features into a single "tags" string for each title
df['tags'] = (df['type_clean'] + ' ' + 
              df['director_clean'] + ' ' + 
              df['country_clean'] + ' ' + 
              df['rating_clean'] + ' ' + 
              df['listed_in_clean'])

# ==========================================
# Step 2: Convert text data into machine-readable format
# ==========================================
# Using TF-IDF to vectorize the tags. This assigns weights to words based on 
# their frequency, penalizing overly common words (like 'movie') and rewarding unique ones.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ==========================================
# Step 3: Calculate content similarity scores
# ==========================================
# Cosine similarity measures the angle between our vectorized features.
# A score of 1 means identical content, 0 means completely different.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ==========================================
# Step 4: Generate recommendations for selected titles
# ==========================================
# Create a reverse mapping of titles to their DataFrame indices for fast lookup
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, n_recommendations=5):
    """Returns the top N similar titles based on content similarity."""
    if title not in indices:
        return f"Title '{title}' not found in the dataset."
    
    # Get the index of the requested title
    idx = indices[title]
    
    # Retrieve the pairwise similarity scores for this index
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the titles based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top N most similar movies (ignoring the 0th item, which is the movie itself)
    sim_scores = sim_scores[1:n_recommendations + 1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return a DataFrame of the recommended titles
    return df.iloc[movie_indices][['title', 'type', 'listed_in', 'director']]

# ==========================================
# Step 5: Evaluate recommendation quality
# ==========================================
print("Recommendations for 'Midnight Mass' (TV Horror/Mystery):")
print(get_recommendations('Midnight Mass'))

print("\nRecommendations for 'Sankofa' (Independent Drama):")
print(get_recommendations('Sankofa'))