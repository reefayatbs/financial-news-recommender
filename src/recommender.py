import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import pickle
from datetime import datetime

class NewsRecommender:
    def __init__(self):
        self.articles_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.user_preferences = {}
        self.user_profiles = {}
        self.is_trained = False
        
    def load_articles(self, filename: str = 'articles.csv'):
        """Load articles and prepare content features"""
        self.articles_df = pd.read_csv(filename)
        self.articles_df['publishedAt'] = pd.to_datetime(self.articles_df['publishedAt'])
        
    def train_content_model(self):
        """Train the TF-IDF model on article content"""
        if self.articles_df is None:
            raise ValueError("Please load articles first using load_articles()")
            
        # Combine title and description for better feature representation
        content = self.articles_df['title'] + ' ' + self.articles_df['description'].fillna('')
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content)
        self.is_trained = True
        
    def load_user_profiles(self, filename: str = 'user_profiles.csv'):
        """Load user risk tolerance profiles"""
        profiles_df = pd.read_csv(filename)
        for _, row in profiles_df.iterrows():
            user_id = row['user_id']
            risk_tolerance = row['risk_tolerance']
            
            # Set interaction weights based on risk tolerance
            if risk_tolerance == 'conservative':
                weights = {'view': 0.7, 'like': 0.2, 'bookmark': 0.1}
            elif risk_tolerance == 'moderate':
                weights = {'view': 0.5, 'like': 0.3, 'bookmark': 0.2}
            else:  # aggressive
                weights = {'view': 0.4, 'like': 0.4, 'bookmark': 0.2}
            
            # Normalize weights
            total = sum(weights.values())
            scale_factor = 5
            weights = {k: (v/total) * scale_factor for k, v in weights.items()}
            
            self.user_profiles[user_id] = {
                'risk_tolerance': risk_tolerance,
                'interaction_weights': weights
            }

    def save_trained_model(self, filename: str = 'trained_recommender.pkl'):
        """Save only the trained components (TF-IDF, content features)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'articles_df': self.articles_df
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_trained_model(self, filename: str = 'trained_recommender.pkl'):
        """Load the trained components"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.articles_df = model_data['articles_df']
        self.is_trained = True

    def update_user_preferences(self, user_id: str, article_id: int, interaction_type: str):
        """Update user preferences based on their interactions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before updating preferences")
            
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = []
        
        if user_id in self.user_profiles:
            weight = self.user_profiles[user_id]['interaction_weights'].get(interaction_type, 1)
        else:
            default_weights = {'view': 1, 'like': 3, 'bookmark': 5}
            weight = default_weights.get(interaction_type, 1)
        
        self.user_preferences[user_id].append({
            'article_id': article_id,
            'weight': weight,
            'interaction_type': interaction_type,
            'timestamp': datetime.now()
        })

    def calculate_article_similarity(self, article_id1: int, article_id2: int) -> float:
        """Calculate similarity between two articles"""
        vec1 = self.tfidf_matrix[article_id1]
        vec2 = self.tfidf_matrix[article_id2]
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity
        
    def get_article_similarities(self, article_id: int, user_interactions: List[Dict]) -> List[Tuple[int, float, str]]:
        """Calculate similarities between an article and user's interaction history"""
        similarities = []
        for interaction in user_interactions:
            interacted_id = interaction['article_id']
            similarity = self.calculate_article_similarity(article_id, interacted_id)
            similarities.append((interacted_id, similarity, interaction['interaction_type']))
        return sorted(similarities, key=lambda x: x[1], reverse=True)
        
    def get_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict]:
        """
        Get personalized article recommendations for a user
        
        Args:
            user_id (str): Unique identifier for the user
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Dict]: List of recommended articles with their details and similarity metrics
        """
        if user_id not in self.user_preferences:
            # For new users, return most recent articles
            recommendations = self.articles_df.sort_values('publishedAt', ascending=False)
            return recommendations.head(n_recommendations).to_dict('records')
        
        # Calculate user profile based on their interactions
        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        total_weight = 0
        
        for interaction in self.user_preferences[user_id]:
            article_id = interaction['article_id']
            weight = interaction['weight']
            article_vector = self.tfidf_matrix[article_id].toarray().flatten()
            user_profile += article_vector * weight
            total_weight += weight
            
        user_profile = user_profile / total_weight
        
        # Calculate similarity between user profile and all articles
        article_similarities = cosine_similarity(
            user_profile.reshape(1, -1),
            self.tfidf_matrix
        ).flatten()
        
        # Get indices of top recommendations
        already_interacted = {
            inter['article_id'] for inter in self.user_preferences[user_id]
        }
        
        recommendation_indices = []
        recommendation_scores = []
        
        for idx in article_similarities.argsort()[::-1]:
            if idx not in already_interacted:
                # Calculate similarities with interacted articles
                similarities = self.get_article_similarities(idx, self.user_preferences[user_id])
                avg_similarity = np.mean([sim[1] for sim in similarities])
                most_similar = max(similarities, key=lambda x: x[1]) if similarities else (None, 0, None)
                
                recommendation_indices.append(idx)
                recommendation_scores.append({
                    'profile_similarity': article_similarities[idx],
                    'avg_interaction_similarity': avg_similarity,
                    'most_similar_article': {
                        'article_id': most_similar[0],
                        'similarity': most_similar[1],
                        'interaction_type': most_similar[2]
                    } if most_similar[0] is not None else None
                })
                
                if len(recommendation_indices) >= n_recommendations:
                    break
        
        # Get recommended articles
        recommendations = []
        for idx, scores in zip(recommendation_indices, recommendation_scores):
            article = self.articles_df.iloc[idx].to_dict()
            article.update({
                'similarity_metrics': scores
            })
            recommendations.append(article)
            
        return recommendations

if __name__ == "__main__":
    # Example usage
    recommender = NewsRecommender()
    recommender.load_articles()
    recommender.load_user_profiles()  # Load user profiles
    
    # Simulate user interactions
    user_id = "user_1"
    recommender.update_user_preferences(user_id, 0, 'like')
    recommender.update_user_preferences(user_id, 1, 'bookmark')
    
    # Get recommendations
    recommendations = recommender.get_recommendations(user_id)
    for rec in recommendations:
        print(f"Title: {rec['title']}")
        print(f"Source: {rec['source']}")
        print(f"URL: {rec['url']}")
        print(f"Profile Similarity: {rec['similarity_metrics']['profile_similarity']:.2f}")
        print(f"Avg Interaction Similarity: {rec['similarity_metrics']['avg_interaction_similarity']:.2f}")
        if rec['similarity_metrics']['most_similar_article']['article_id'] is not None:
            similar_article = recommender.articles_df.iloc[rec['similarity_metrics']['most_similar_article']['article_id']]['title']
            print(f"Most Similar Article: {similar_article}")
            print(f"Similarity Score: {rec['similarity_metrics']['most_similar_article']['similarity']:.2f}")
            print(f"Interaction Type: {rec['similarity_metrics']['most_similar_article']['interaction_type']}")
        print("---") 