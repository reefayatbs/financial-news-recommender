import os
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
import time
from tqdm import tqdm

def fetch_articles():
    """Fetch financial news articles from NewsAPI"""
    load_dotenv()
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key:
        raise ValueError("Please set NEWS_API_KEY in your .env file")
    
    newsapi = NewsApiClient(api_key=api_key)
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get last 30 days of news
    
    # Optimized search queries combining related terms
    queries = [
        'stock market OR financial markets',
        'cryptocurrency OR blockchain',
        'trading OR investment',
        'finance technology OR fintech',
        'market analysis OR trading strategy',
        'economic news OR business news',
        'tech stocks OR technology companies'
    ]
    
    # Financial and business news sources
    domains = [
        'bloomberg.com,reuters.com,ft.com,wsj.com,cnbc.com,'
        'businessinsider.com,forbes.com,marketwatch.com,'
        'investing.com,finance.yahoo.com'
    ]
    
    all_articles = []
    
    print("\nFetching articles from NewsAPI...")
    progress_bar = tqdm(queries, desc="Processing queries")
    
    # Fetch articles for each query
    for query in progress_bar:
        try:
            # Search everything
            articles = newsapi.get_everything(
                q=query,
                domains=domains[0],
                language='en',
                sort_by='relevancy',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                page_size=100  # Maximum articles per request
            )
            
            if articles['status'] == 'ok':
                all_articles.extend(articles['articles'])
                progress_bar.set_postfix({"Articles": len(articles['articles'])})
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\nError fetching articles for query '{query}': {str(e)}")
            if "rateLimited" in str(e):
                print("Rate limit reached. Waiting for 2 seconds...")
                time.sleep(2)
            continue
    
    print("\nProcessing fetched articles...")
    
    # Remove duplicates based on URL
    unique_articles = []
    seen_urls = set()
    
    for article in all_articles:
        if article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            unique_articles.append(article)
    
    # Convert to DataFrame
    df = pd.DataFrame(unique_articles)
    
    if df.empty:
        print("No articles were fetched. Please check your API key and try again.")
        return pd.DataFrame()
    
    # Clean and prepare data
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['source'] = df['source'].apply(lambda x: x['name'])
    
    # Sort by published date
    df = df.sort_values('publishedAt', ascending=False)
    
    # Save to CSV
    df.to_csv('articles.csv', index=True)
    print(f"\nSaved {len(df)} unique articles to articles.csv")
    
    # Print some statistics
    print("\nArticle Statistics:")
    print(f"Date Range: {df['publishedAt'].min().date()} to {df['publishedAt'].max().date()}")
    print("\nTop Sources:")
    print(df['source'].value_counts().head())
    
    return df

if __name__ == "__main__":
    fetch_articles() 