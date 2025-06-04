print("Script starting...")

import os
import subprocess
from src.news_fetcher import fetch_articles
from src.user_simulator import UserSimulator
from src.recommender import NewsRecommender
import pandas as pd

def main(fetch_new=False):
    print("Starting system...")
    
    # 1. Get articles (either from file or fetch new ones)
    if fetch_new:
        print("Fetching new articles...")
        articles_df = fetch_articles()
    else:
        print("Loading stored articles from articles.csv...")
        try:
            articles_df = pd.read_csv('articles.csv')
            print(f"Successfully loaded {len(articles_df)} articles from storage")
        except FileNotFoundError:
            print("No stored articles found. Fetching new articles...")
            articles_df = fetch_articles()
    
    # 2. Train the recommender system on article content
    print("\nInitializing recommender system...")
    recommender = NewsRecommender()
    print("Loading articles into recommender...")
    recommender.load_articles('articles.csv')
    print("Training content model (TF-IDF)...")
    recommender.train_content_model()  # Train TF-IDF and content features
    
    # Save the trained model (content features only)
    print("Saving trained model...")
    recommender.save_trained_model()
    print("Model saved successfully!")
    
    # 3. Generate some training interactions (for demonstration)
    print("\nInitializing user simulator...")
    simulator = UserSimulator(articles_df, n_users=10)
    print("Generating user interactions...")
    simulator.generate_interactions(days=30)
    print("Saving interactions and profiles...")
    simulator.save_interactions()
    simulator.save_user_profiles()
    print("Simulation complete!")
    
    # 4. Launch the dashboard
    print("\nLaunching dashboard...")
    subprocess.run(["streamlit", "run", "src/dashboard.py"])

if __name__ == "__main__":
    main(fetch_new=False)  # Set to True if you want to fetch new articles 