import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading spaCy model: {str(e)}")
    nlp = None

# Download and verify required NLTK data
def download_nltk_data():
    """Download required NLTK data and verify its availability"""
    required_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    try:
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                print(f"Downloading {package}...")
                nltk.download(package, quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
        raise

# Ensure NLTK data is available
download_nltk_data()

class UserExperienceAnalyzer:
    def __init__(self, source):
        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
        else:
            self.df = pd.read_csv(source)
        # Filter for non-bug experiences
        self.df = self.df[self.df['is_bug_report'] == 0].copy()
        self.df['review_date'] = pd.to_datetime(self.df['review_date'])
        self.preprocess_text()
        
    def preprocess_text(self):
        """Preprocess the review text data"""
        try:
            self.df['cleaned_text'] = self.df['review_description'].fillna('')
            self.df['cleaned_text'] = self.df['cleaned_text'].apply(self.clean_text)
            self.df['sentiment_score'] = self.df['cleaned_text'].apply(self.get_sentiment_score)
            self.df['sentiment_category'] = pd.cut(
                self.df['sentiment_score'],
                bins=[-1, -0.3, 0.3, 1],
                labels=['Negative', 'Neutral', 'Positive']            )
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            raise
            
    def clean_text(self, text):
        """Clean and preprocess text"""
        try:
            if pd.isna(text):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            try:
                # Tokenize
                tokens = word_tokenize(text)
            except Exception as e:
                print(f"Tokenization error: {str(e)}")
                # Fallback to simple space-based tokenization
                tokens = text.split()
            
            try:
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
            except Exception as e:
                print(f"Stopwords error: {str(e)}")
                # Continue with unfiltered tokens
                
            try:
                # Lemmatization
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            except Exception as e:
                print(f"Lemmatization error: {str(e)}")
                # Continue with unlemmatized tokens
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Text cleaning error: {str(e)}")
            return text  # Return original text if processing fails
            
    def get_sentiment_score(self, text):
        """Get sentiment score using TextBlob"""
        return TextBlob(text).sentiment.polarity
    
    def analyze_sentiment_trends(self):
        """Analyze sentiment trends over time"""
        monthly_sentiment = self.df.set_index('review_date').resample('ME')[['sentiment_score']].mean()
        monthly_count = self.df.set_index('review_date').resample('ME').size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_sentiment.index,
            y=monthly_sentiment['sentiment_score'],
            name='Average Sentiment',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Bar(
            x=monthly_count.index,
            y=monthly_count,
            name='Review Count',
            yaxis='y2',
            opacity=0.3
        ))
        
        fig.update_layout(
            title='Monthly Sentiment Trends and Review Volume',
            yaxis=dict(title='Sentiment Score'),
            yaxis2=dict(title='Review Count', overlaying='y', side='right'),
            hovermode='x'
        )
        return fig
    
    def analyze_version_sentiment(self):
        """Analyze sentiment across app versions"""
        # Group by appVersion and calculate metrics
        version_sentiment = self.df.groupby('appVersion').agg({
            'sentiment_score': 'mean',
            'rating': 'mean',
            'review_description': 'count'
        }).reset_index()
        
        version_sentiment.columns = ['appVersion', 'sentiment_mean', 'rating_mean', 'review_count']
        version_sentiment = version_sentiment.dropna(subset=['appVersion'])
        
        fig = px.scatter(
            version_sentiment,
            x='appVersion',
            y='sentiment_mean',
            size='review_count',
            color='rating_mean',
            title='Sentiment Analysis by App Version',
            labels={
                'appVersion': 'App Version',
                'sentiment_mean': 'Average Sentiment',
                'rating_mean': 'Average Rating',
                'review_count': 'Number of Reviews'
            },
            color_continuous_scale='RdYlGn'  # Red to Yellow to Green color scale
        )
        return fig
    
    def extract_key_phrases(self):
        """Extract and analyze key phrases from reviews"""
        # Use TF-IDF to identify important terms
        tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(self.df['cleaned_text'])
        
        # Get feature names and their scores
        feature_names = tfidf.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        
        # Create DataFrame of terms and their importance
        key_phrases = pd.DataFrame({
            'term': feature_names,
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        return key_phrases.head(20)
    
    def create_topic_clusters(self, n_topics=5):
        """Create topic clusters using LDA"""
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(self.df['cleaned_text'])
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = lda.fit_transform(tfidf_matrix)
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        # Create topic-word matrix
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics[f'Topic {topic_idx + 1}'] = top_words
        
        return pd.DataFrame.from_dict(topics, orient='index')
    
    def generate_wordcloud(self):
        """Generate word cloud from reviews"""
        text = ' '.join(self.df['cleaned_text'])
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100
        ).generate(text)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig

    def get_sentiment_summary(self):
        """Get summary statistics of sentiment analysis"""
        summary = {
            'total_reviews': len(self.df),
            'average_sentiment': self.df['sentiment_score'].mean(),
            'sentiment_distribution': self.df['sentiment_category'].value_counts().to_dict(),
            'average_rating': self.df['rating'].mean(),
            'rating_distribution': self.df['rating'].value_counts().to_dict()
        }
        return summary
