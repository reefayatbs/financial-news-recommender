import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from recommender import NewsRecommender
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def load_data():
    """Load all necessary data files"""
    articles_df = pd.read_csv('articles.csv')
    interactions_df = pd.read_csv('user_interactions.csv')
    profiles_df = pd.read_csv('user_profiles.csv')
    
    # Convert timestamp to datetime
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt'])
    
    return articles_df, interactions_df, profiles_df

def extract_top_keywords(text: str, vectorizer: TfidfVectorizer, n_keywords: int = 5) -> list:
    """Extract top keywords from text using the trained vectorizer"""
    # Transform the text
    vector = vectorizer.transform([text])
    
    # Get feature names and their TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    scores = vector.toarray()[0]
    
    # Get top keywords
    top_indices = scores.argsort()[-n_keywords:][::-1]
    return [(feature_names[i], round(scores[i], 3)) for i in top_indices if scores[i] > 0]

def get_user_top_keywords(user_interactions: pd.DataFrame, articles_df: pd.DataFrame, vectorizer: TfidfVectorizer, n_keywords: int = 10):
    """Get top keywords from user's interaction history"""
    # Combine all interacted article content
    user_content = []
    for _, interaction in user_interactions.iterrows():
        article = articles_df.iloc[interaction['article_id']]
        content = f"{article['title']} {article['description']}"
        # Weight content by interaction type
        weight = 3 if interaction['interaction_type'] == 'bookmark' else 2 if interaction['interaction_type'] == 'like' else 1
        user_content.extend([content] * weight)
    
    combined_content = " ".join(user_content)
    return extract_top_keywords(combined_content, vectorizer, n_keywords)

def plot_interaction_types(interactions_df, user_id):
    """Plot distribution of interaction types"""
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    interaction_counts = user_interactions['interaction_type'].value_counts()
    
    fig = px.pie(values=interaction_counts.values, 
                 names=interaction_counts.index,
                 title=f'User {user_id} Interaction Types')
    
    # Adjust layout to prevent overlapping
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

def plot_source_distribution(interactions_df, user_id):
    """Plot distribution of news sources"""
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    source_counts = user_interactions['article_source'].value_counts()
    
    fig = px.bar(x=source_counts.index, 
                 y=source_counts.values,
                 title=f'User {user_id} News Source Distribution')
    
    # Adjust layout to prevent overlapping
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Source",
        yaxis_title="Number of Interactions",
        xaxis=dict(tickangle=45)
    )
    return fig

def plot_topic_distribution(interactions_df, articles_df, user_id):
    """Plot distribution of article topics/keywords"""
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    # Extract topics from article titles and descriptions
    topics = []
    for _, interaction in user_interactions.iterrows():
        article = articles_df.iloc[interaction['article_id']]
        text = f"{article['title']} {article['description']}"
        text = text.lower()
        
        # Check for common financial topics
        if 'stock' in text or 'market' in text:
            topics.append('Stock Market')
        elif 'crypto' in text or 'bitcoin' in text:
            topics.append('Cryptocurrency')
        elif 'tech' in text or 'technology' in text:
            topics.append('Technology')
        elif 'trade' in text or 'trading' in text:
            topics.append('Trading')
        elif 'invest' in text:
            topics.append('Investment')
        else:
            topics.append('Other')
    
    topic_counts = pd.Series(topics).value_counts()
    
    fig = px.bar(x=topic_counts.index, 
                 y=topic_counts.values,
                 title=f'User {user_id} Topic Distribution')
    
    # Adjust layout to prevent overlapping
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Topic",
        yaxis_title="Number of Articles"
    )
    return fig

def display_keyword_comparison(article_keywords, user_keywords):
    """Display keyword comparison between article and user interests"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Article Keywords")
        for keyword, score in article_keywords:
            st.markdown(f"- {keyword} ({score:.3f})")
    
    with col2:
        st.markdown("#### User Interest Keywords")
        for keyword, score in user_keywords:
            st.markdown(f"- {keyword} ({score:.3f})")

def main():
    st.title('News Article Recommender Dashboard')
    
    # Load data
    articles_df, interactions_df, profiles_df = load_data()
    
    # Initialize recommender
    recommender = NewsRecommender()
    recommender.load_articles()
    recommender.train_content_model()
    recommender.load_user_profiles()
    
    # Process current session's interactions
    for _, interaction in interactions_df.iterrows():
        recommender.update_user_preferences(
            str(interaction['user_id']),
            interaction['article_id'],
            interaction['interaction_type']
        )
    
    # Sidebar - User selection
    users = sorted(interactions_df['user_id'].unique())
    selected_user = st.sidebar.selectbox('Select User', users)
    
    # Display user profile
    st.subheader('User Profile')
    user_profile = profiles_df[profiles_df['user_id'] == selected_user].iloc[0]
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric('Interests', user_profile['interests'])
    col2.metric('Activity Level', user_profile['activity_level'])
    col3.metric('Risk Tolerance', user_profile['risk_tolerance'])
    
    # Add spacing
    st.markdown("---")
    
    # User Interaction Analysis
    st.subheader('User Interaction Analysis')
    
    # Interaction Types and Source Distribution
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('Interaction Types')
        pie_fig = plot_interaction_types(interactions_df, selected_user)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        st.subheader('News Sources')
        source_fig = plot_source_distribution(interactions_df, selected_user)
        st.plotly_chart(source_fig, use_container_width=True)
    
    # Add spacing
    st.markdown("---")
    
    # Topic Distribution
    st.subheader('Topic Distribution')
    topic_fig = plot_topic_distribution(interactions_df, articles_df, selected_user)
    st.plotly_chart(topic_fig, use_container_width=True)
    
    # Add spacing
    st.markdown("---")
    
    # Get user's interaction history keywords
    user_interactions = interactions_df[interactions_df['user_id'] == selected_user]
    user_keywords = get_user_top_keywords(user_interactions, articles_df, recommender.tfidf_vectorizer)
    
    # Recommendations with keyword comparison
    st.subheader('Recommendations')
    recommendations = recommender.get_recommendations(selected_user, n_recommendations=5)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"### {i}. {rec['title']}")
        st.markdown(f"Source: {rec['source']}")
        st.markdown(f"[Read Article]({rec['url']})")
        
        # Extract keywords from recommended article
        article_content = f"{rec['title']} {rec['description']}"
        article_keywords = extract_top_keywords(article_content, recommender.tfidf_vectorizer)
        
        # Display keyword comparison
        display_keyword_comparison(article_keywords, user_keywords)
        st.markdown("---")

if __name__ == '__main__':
    main() 