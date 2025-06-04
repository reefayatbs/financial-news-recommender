# Financial News Article Recommender System

A content-based recommender system for financial news articles that personalizes recommendations based on user interactions and risk tolerance profiles.

## Overview

This system fetches financial news articles and provides personalized recommendations based on:
- Content similarity using TF-IDF vectorization
- User interaction patterns (views, likes, bookmarks)
- Risk tolerance profiles (conservative, moderate, aggressive)

## System Components

### 1. News Fetcher (`src/news_fetcher.py`)
- Fetches financial news from NewsAPI
- Optimized queries for financial topics
- Rate limit handling and progress tracking
- Focused on core financial news sources

### 2. Recommender Engine (`src/recommender.py`)
- Content-based filtering using TF-IDF
- Personalized weighting based on risk profiles:
  - Conservative: views (0.7), likes (0.2), bookmarks (0.1)
  - Moderate: views (0.5), likes (0.3), bookmarks (0.2)
  - Aggressive: views (0.4), likes (0.4), bookmarks (0.2)
- Similarity metrics for recommendation explanations

### 3. User Simulator (`src/user_simulator.py`)
- Generates synthetic user interactions
- Creates diverse user profiles
- Simulates realistic interaction patterns

### 4. Dashboard (`src/dashboard.py`)
- Interactive Streamlit dashboard
- Visualization of user interactions
- Real-time recommendation display
- Detailed similarity metrics

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up NewsAPI:
- Get an API key from [NewsAPI](https://newsapi.org/)
- Create a `.env` file in the project root
- Add your API key: `NEWS_API_KEY=your_api_key_here`

## Usage

1. Run the complete system:
```bash
python run_system.py
```

2. To fetch new articles:
```bash
python run_system.py --fetch-new
```

3. Access the dashboard:
- Open browser to `http://localhost:8501`
- Select a user from the sidebar
- View recommendations and interaction analytics

## Technical Details

### Content Processing
- TF-IDF vectorization of article content
- N-gram range: (1, 2)
- Stop words removed
- Maximum 5000 features

### Similarity Metrics
- Cosine similarity between article vectors

### Recommendation Process
1. Build user profile from weighted interactions
2. Calculate content similarity using TF-IDF vectors
3. Filter out previously interacted articles
4. Rank by similarity and user preferences
5. Add explanation metrics

## Limitations and Future Improvements

1. Cold Start
- New users receive recent articles
- Could be improved with explicit preferences

2. Content Updates
- Model needs retraining for new articles
- Could implement incremental batch updates

3. Evaluation
- Current evaluation uses simulated data
- Could implement A/B testing
- Need train/test split for better accuracy measurement

## Dependencies

- pandas>=2.2.0
- numpy>=1.26.0
- scikit-learn>=1.4.0
- newsapi-python==0.2.7
- python-dotenv==1.0.0
- streamlit==1.32.0
- plotly==5.19.0
- tqdm>=4.66.0
