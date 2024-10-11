import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample user-item ratings data
ratings_data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4],
    'item_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'A', 'D'],
    'rating': [5, 4, 3, 4, 5, 5, 2, 4, 1]
}

# Sample item descriptions for content-based filtering
items_data = {
    'item_id': ['A', 'B', 'C', 'D'],
    'description': [
        'Action movie with adventure',
        'Romantic comedy with drama',
        'Action thriller with suspense',
        'Historical documentary'
    ]
}

# Create DataFrames
ratings_df = pd.DataFrame(ratings_data)
items_df = pd.DataFrame(items_data)

# Collaborative Filtering
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_collaborative(user_id, n_recommendations=2):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    recommendations = pd.Series(dtype=float)  # Initialize an empty Series
    
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        recommendations = pd.concat([recommendations, similar_user_ratings[similar_user_ratings > 0]])
    
    recommendations = recommendations.groupby(recommendations.index).mean().sort_values(ascending=False)
    return recommendations.head(n_recommendations)

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(items_df['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_content(item_id, n_recommendations=2):
    idx = items_df.index[items_df['item_id'] == item_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations + 1]
    item_indices = [i[0] for i in sim_scores]
    return items_df['item_id'].iloc[item_indices]

# Example usage
user_id = 1
print(f"Collaborative recommendations for user {user_id}:")
print(recommend_collaborative(user_id))

item_id = 'A'
print(f"\nContent-based recommendations for item {item_id}:")
print(recommend_content(item_id))
