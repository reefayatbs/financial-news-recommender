import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict

class UserSimulator:
    def __init__(self, articles_df: pd.DataFrame, n_users: int = 10):
        self.articles_df = articles_df
        self.n_users = n_users
        self.user_profiles = self._generate_user_profiles()
        self.interactions = []
        
    def _generate_user_profiles(self) -> Dict[str, Dict]:
        """Generate simulated user profiles with interests"""
        profiles = {}
        interests = [
            'technology', 'finance', 'cryptocurrency', 'stocks',
            'commodities', 'forex', 'real estate', 'startups',
            'market analysis', 'trading strategies', 'investment',
            'blockchain', 'artificial intelligence', 'fintech'
        ]
        
        risk_profiles = {
            'conservative': {'view_ratio': 0.7, 'like_ratio': 0.2, 'bookmark_ratio': 0.1},
            'moderate': {'view_ratio': 0.5, 'like_ratio': 0.3, 'bookmark_ratio': 0.2},
            'aggressive': {'view_ratio': 0.4, 'like_ratio': 0.4, 'bookmark_ratio': 0.2}
        }
        
        activity_levels = {
            'low': {'daily_interactions': (2, 5)},
            'medium': {'daily_interactions': (5, 10)},
            'high': {'daily_interactions': (10, 20)}
        }
        
        for i in range(self.n_users):
            user_id = f"user_{i+1}"
            # Each user gets 3-6 random interests
            user_interests = random.sample(interests, random.randint(3, 6))
            risk_tolerance = random.choice(list(risk_profiles.keys()))
            activity_level = random.choice(list(activity_levels.keys()))
            
            profiles[user_id] = {
                'interests': user_interests,
                'activity_level': activity_level,
                'risk_tolerance': risk_tolerance,
                'interaction_ratios': risk_profiles[risk_tolerance],
                'daily_range': activity_levels[activity_level]['daily_interactions']
            }
        
        return profiles
    
    def _calculate_article_relevance(self, article: pd.Series, user_interests: List[str]) -> float:
        """Calculate how relevant an article is to a user's interests"""
        text = f"{article['title']} {article['description']}"
        text = text.lower()
        
        # Calculate weighted relevance based on presence of interest keywords
        relevance = 0
        for interest in user_interests:
            # Check for exact matches and partial matches
            if interest in text:
                relevance += 1
            # Check for related terms
            related_terms = {
                'technology': ['tech', 'software', 'digital'],
                'finance': ['financial', 'money', 'banking'],
                'cryptocurrency': ['crypto', 'bitcoin', 'blockchain'],
                'stocks': ['stock market', 'shares', 'equity'],
                'trading': ['trader', 'market', 'exchange']
            }
            if interest in related_terms:
                for term in related_terms[interest]:
                    if term in text:
                        relevance += 0.5
        
        return relevance / (len(user_interests) * 1.5)  # Normalize score
    
    def _generate_daily_interactions(self, user_id: str, date: datetime, available_articles: List[int]) -> List[Dict]:
        """Generate interactions for a single day for a user"""
        profile = self.user_profiles[user_id]
        min_daily, max_daily = profile['daily_range']
        n_interactions = random.randint(min_daily, max_daily)
        
        # Get interaction ratios
        ratios = profile['interaction_ratios']
        
        daily_interactions = []
        selected_articles = set()  # Track articles already interacted with
        
        # Sort articles by relevance
        article_relevance = []
        for idx in available_articles:
            if idx not in selected_articles:
                relevance = self._calculate_article_relevance(self.articles_df.iloc[idx], profile['interests'])
                article_relevance.append((idx, relevance))
        
        article_relevance.sort(key=lambda x: x[1], reverse=True)
        top_articles = [idx for idx, _ in article_relevance[:n_interactions*2]]  # Get twice as many articles as needed
        
        for _ in range(n_interactions):
            if not top_articles:
                break
                
            article_idx = random.choice(top_articles)
            top_articles.remove(article_idx)
            selected_articles.add(article_idx)
            
            # Determine interaction type based on ratios and randomness
            rand = random.random()
            if rand < ratios['view_ratio']:
                interaction_type = 'view'
            elif rand < ratios['view_ratio'] + ratios['like_ratio']:
                interaction_type = 'like'
            else:
                interaction_type = 'bookmark'
            
            # Generate timestamp within the day
            interaction_time = date + timedelta(
                hours=random.randint(8, 22),  # Business hours
                minutes=random.randint(0, 59)
            )
            
            daily_interactions.append({
                'user_id': user_id,
                'article_id': article_idx,
                'interaction_type': interaction_type,
                'timestamp': interaction_time,
                'article_title': self.articles_df.iloc[article_idx]['title'],
                'article_source': self.articles_df.iloc[article_idx]['source']
            })
        
        return daily_interactions
    
    def generate_interactions(self, days: int = 30) -> List[Dict]:
        """Generate user interactions over a period of time"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get list of available article indices
        available_articles = list(range(len(self.articles_df)))
        
        for user_id in self.user_profiles:
            current_date = start_date
            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                    daily_interactions = self._generate_daily_interactions(
                        user_id, current_date, available_articles
                    )
                    self.interactions.extend(daily_interactions)
                current_date += timedelta(days=1)
        
        # Sort interactions by timestamp
        self.interactions.sort(key=lambda x: x['timestamp'])
        return self.interactions
    
    def save_interactions(self, filename: str = 'user_interactions.csv'):
        """Save interactions to CSV file"""
        df = pd.DataFrame(self.interactions)
        df.to_csv(filename, index=False)
        print(f"Saved {len(self.interactions)} interactions to {filename}")
        
    def save_user_profiles(self, filename: str = 'user_profiles.csv'):
        """Save user profiles to CSV file"""
        profiles_list = []
        for user_id, profile in self.user_profiles.items():
            profile_dict = {
                'user_id': user_id,
                'interests': ','.join(profile['interests']),
                'activity_level': profile['activity_level'],
                'risk_tolerance': profile['risk_tolerance']
            }
            profiles_list.append(profile_dict)
        
        df = pd.DataFrame(profiles_list)
        df.to_csv(filename, index=False)
        print(f"Saved {len(self.user_profiles)} user profiles to {filename}")

if __name__ == "__main__":
    # Load articles
    articles_df = pd.read_csv('articles.csv')
    
    # Create simulator and generate interactions
    simulator = UserSimulator(articles_df, n_users=10)
    interactions = simulator.generate_interactions(days=30)
    
    # Save data
    simulator.save_interactions()
    simulator.save_user_profiles() 