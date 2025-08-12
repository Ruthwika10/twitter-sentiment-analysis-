import csv
import re
import json
import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class AdvancedSentimentAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.tweets_data = []
        self.processed_tweets = []
        self.sentiment_scores = []
        self.keywords = []
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Sentiment categories with refined thresholds
        self.sentiment_categories = {
            'Extremely Positive': (0.8, 1.0),
            'Very Positive': (0.6, 0.8),
            'Positive': (0.3, 0.6),
            'Slightly Positive': (0.1, 0.3),
            'Neutral': (-0.1, 0.1),
            'Slightly Negative': (-0.3, -0.1),
            'Negative': (-0.6, -0.3),
            'Very Negative': (-0.8, -0.6),
            'Extremely Negative': (-1.0, -0.8)
        }
        
        # Color scheme for visualizations
        self.colors = {
            'Extremely Positive': '#006400',
            'Very Positive': '#228B22',
            'Positive': '#32CD32',
            'Slightly Positive': '#90EE90',
            'Neutral': '#808080',
            'Slightly Negative': '#FFA500',
            'Negative': '#FF6347',
            'Very Negative': '#DC143C',
            'Extremely Negative': '#8B0000'
        }

    def load_and_preprocess_data(self):
        """Load data from CSV and perform initial preprocessing"""
        print("Loading and preprocessing data...")
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            for i, row in enumerate(reader):
                if row and len(row) >= 3:
                    # Assume columns: [id, timestamp, text, ...]
                    tweet_data = {
                        'id': i,
                        'timestamp': row[1] if len(row) > 1 else f"2024-01-{(i%30)+1:02d}",
                        'text': row[2] if len(row) > 2 else row[0],
                        'user': row[3] if len(row) > 3 else f"user_{i}",
                        'location': row[4] if len(row) > 4 else "Unknown"
                    }
                    self.tweets_data.append(tweet_data)
        
        print(f"Loaded {len(self.tweets_data)} tweets")
        return len(self.tweets_data)

    def clean_text(self, text):
        """Advanced text cleaning with multiple preprocessing steps"""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
        
        # Convert to lowercase and remove extra whitespace
        text = ' '.join(text.lower().split())
        
        return text

    def extract_features(self, text):
        """Extract linguistic features from text"""
        features = {}
        
        # Basic metrics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len([c for c in text if c.isalpha()])
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_count'] = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = features['caps_count'] / len(text) if len(text) > 0 else 0
        
        # Emotional indicators
        positive_words = ['great', 'awesome', 'amazing', 'fantastic', 'wonderful', 'excellent', 'love', 'best', 'perfect', 'brilliant']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting', 'pathetic', 'useless', 'stupid', 'annoying']
        
        features['positive_word_count'] = sum(1 for word in positive_words if word in text.lower())
        features['negative_word_count'] = sum(1 for word in negative_words if word in text.lower())
        
        return features

    def analyze_sentiments(self, keyword=None, num_tweets=None):
        """Perform comprehensive sentiment analysis"""
        print("Performing sentiment analysis...")
        
        # Filter tweets by keyword if provided
        filtered_tweets = self.tweets_data
        if keyword:
            keyword = keyword.lower()
            filtered_tweets = [tweet for tweet in self.tweets_data 
                             if keyword in tweet['text'].lower()]
            print(f"Found {len(filtered_tweets)} tweets containing '{keyword}'")
        
        if num_tweets:
            filtered_tweets = filtered_tweets[:num_tweets]
        
        if not filtered_tweets:
            print("No tweets found matching criteria.")
            return
        
        # Process each tweet
        for tweet in filtered_tweets:
            clean_text = self.clean_text(tweet['text'])
            features = self.extract_features(tweet['text'])
            
            # Sentiment analysis
            blob = TextBlob(clean_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Categorize sentiment
            sentiment_category = 'Neutral'
            for category, (min_val, max_val) in self.sentiment_categories.items():
                if min_val <= polarity <= max_val:
                    sentiment_category = category
                    break
            
            processed_tweet = {
                **tweet,
                'clean_text': clean_text,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment_category': sentiment_category,
                'features': features
            }
            
            self.processed_tweets.append(processed_tweet)
            self.sentiment_scores.append(polarity)
        
        print(f"Analyzed {len(self.processed_tweets)} tweets")

    def generate_basic_statistics(self):
        """Generate comprehensive statistics"""
        if not self.processed_tweets:
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SENTIMENT ANALYSIS REPORT")
        print("="*80)
        
        total_tweets = len(self.processed_tweets)
        
        # Basic sentiment statistics
        avg_polarity = np.mean(self.sentiment_scores)
        std_polarity = np.std(self.sentiment_scores)
        median_polarity = np.median(self.sentiment_scores)
        avg_subjectivity = np.mean([t['subjectivity'] for t in self.processed_tweets])
        
        print(f"\nBasic Statistics:")
        print(f"Total tweets analyzed: {total_tweets}")
        print(f"Average polarity: {avg_polarity:.4f}")
        print(f"Standard deviation: {std_polarity:.4f}")
        print(f"Median polarity: {median_polarity:.4f}")
        print(f"Average subjectivity: {avg_subjectivity:.4f}")
        
        # Sentiment distribution
        sentiment_counts = Counter([t['sentiment_category'] for t in self.processed_tweets])
        
        print(f"\nSentiment Distribution:")
        for category in self.sentiment_categories.keys():
            count = sentiment_counts[category]
            percentage = (count / total_tweets) * 100 if total_tweets > 0 else 0
            print(f"{category}: {count} tweets ({percentage:.2f}%)")
        
        # Feature statistics
        avg_length = np.mean([t['features']['length'] for t in self.processed_tweets])
        avg_word_count = np.mean([t['features']['word_count'] for t in self.processed_tweets])
        
        print(f"\nText Characteristics:")
        print(f"Average text length: {avg_length:.2f} characters")
        print(f"Average word count: {avg_word_count:.2f} words")

    def create_advanced_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.processed_tweets:
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Sentiment Distribution Bar Chart
        ax1 = plt.subplot(4, 3, 1)
        sentiment_counts = Counter([t['sentiment_category'] for t in self.processed_tweets])
        categories = list(self.sentiment_categories.keys())
        counts = [sentiment_counts[cat] for cat in categories]
        colors = [self.colors[cat] for cat in categories]
        
        bars = ax1.bar(range(len(categories)), counts, color=colors)
        ax1.set_xlabel('Sentiment Category')
        ax1.set_ylabel('Number of Tweets')
        ax1.set_title('Sentiment Distribution')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Polarity Score Distribution (Histogram)
        ax2 = plt.subplot(4, 3, 2)
        ax2.hist(self.sentiment_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(self.sentiment_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.sentiment_scores):.3f}')
        ax2.set_xlabel('Polarity Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Polarity Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sentiment vs Subjectivity Scatter Plot
        ax3 = plt.subplot(4, 3, 3)
        polarities = [t['polarity'] for t in self.processed_tweets]
        subjectivities = [t['subjectivity'] for t in self.processed_tweets]
        sentiment_categories = [t['sentiment_category'] for t in self.processed_tweets]
        
        # Create scatter plot with different colors for each sentiment
        for category in self.sentiment_categories.keys():
            mask = [cat == category for cat in sentiment_categories]
            if any(mask):
                cat_polarities = [p for p, m in zip(polarities, mask) if m]
                cat_subjectivities = [s for s, m in zip(subjectivities, mask) if m]
                ax3.scatter(cat_polarities, cat_subjectivities, 
                           label=category, alpha=0.6, s=30, color=self.colors[category])
        
        ax3.set_xlabel('Polarity')
        ax3.set_ylabel('Subjectivity')
        ax3.set_title('Sentiment vs Subjectivity')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Text Length vs Sentiment
        ax4 = plt.subplot(4, 3, 4)
        text_lengths = [t['features']['length'] for t in self.processed_tweets]
        ax4.scatter(text_lengths, polarities, alpha=0.6, c=polarities, cmap='RdYlBu')
        ax4.set_xlabel('Text Length')
        ax4.set_ylabel('Polarity')
        ax4.set_title('Text Length vs Sentiment')
        ax4.grid(True, alpha=0.3)
        
        # 5. Word Count Distribution
        ax5 = plt.subplot(4, 3, 5)
        word_counts = [t['features']['word_count'] for t in self.processed_tweets]
        ax5.hist(word_counts, bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
        ax5.set_xlabel('Word Count')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Word Count Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 6. Sentiment Pie Chart
        ax6 = plt.subplot(4, 3, 6)
        positive_cats = ['Extremely Positive', 'Very Positive', 'Positive', 'Slightly Positive']
        negative_cats = ['Slightly Negative', 'Negative', 'Very Negative', 'Extremely Negative']
        
        pos_count = sum(sentiment_counts[cat] for cat in positive_cats)
        neg_count = sum(sentiment_counts[cat] for cat in negative_cats)
        neu_count = sentiment_counts['Neutral']
        
        sizes = [pos_count, neu_count, neg_count]
        labels = ['Positive', 'Neutral', 'Negative']
        colors_pie = ['#32CD32', '#808080', '#FF6347']
        
        wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct='%1.1f%%', startangle=90)
        ax6.set_title('Overall Sentiment Distribution')
        
        # 7. Sentiment Trends (if timestamp data available)
        ax7 = plt.subplot(4, 3, 7)
        try:
            # Group by date and calculate average sentiment
            daily_sentiment = defaultdict(list)
            for tweet in self.processed_tweets:
                date = tweet['timestamp'][:10] if len(tweet['timestamp']) > 10 else tweet['timestamp']
                daily_sentiment[date].append(tweet['polarity'])
            
            dates = sorted(daily_sentiment.keys())
            avg_sentiments = [np.mean(daily_sentiment[date]) for date in dates]
            
            ax7.plot(range(len(dates)), avg_sentiments, marker='o', linewidth=2, markersize=4)
            ax7.set_xlabel('Time Period')
            ax7.set_ylabel('Average Polarity')
            ax7.set_title('Sentiment Trends Over Time')
            ax7.set_xticks(range(0, len(dates), max(1, len(dates)//5)))
            ax7.set_xticklabels([dates[i] for i in range(0, len(dates), max(1, len(dates)//5))], 
                               rotation=45)
            ax7.grid(True, alpha=0.3)
            ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        except:
            ax7.text(0.5, 0.5, 'Timeline data not available', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Sentiment Trends Over Time')
        
        # 8. Feature Correlation Heatmap
        ax8 = plt.subplot(4, 3, 8)
        feature_data = []
        for tweet in self.processed_tweets:
            feature_row = [
                tweet['polarity'],
                tweet['subjectivity'],
                tweet['features']['length'],
                tweet['features']['word_count'],
                tweet['features']['caps_ratio'],
                tweet['features']['exclamation_count'],
                tweet['features']['question_count']
            ]
            feature_data.append(feature_row)
        
        feature_df = pd.DataFrame(feature_data, columns=[
            'Polarity', 'Subjectivity', 'Text Length', 'Word Count',
            'Caps Ratio', 'Exclamations', 'Questions'
        ])
        
        correlation_matrix = feature_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax8, fmt='.2f')
        ax8.set_title('Feature Correlation Matrix')
        
        # 9. Top Positive/Negative Words
        ax9 = plt.subplot(4, 3, 9)
        all_words = []
        positive_words = []
        negative_words = []
        
        for tweet in self.processed_tweets:
            words = tweet['clean_text'].split()
            all_words.extend(words)
            if tweet['polarity'] > 0.3:
                positive_words.extend(words)
            elif tweet['polarity'] < -0.3:
                negative_words.extend(words)
        
        # Remove stop words
        filtered_words = [word for word in all_words if word not in self.stop_words and len(word) > 2]
        word_freq = Counter(filtered_words)
        
        if word_freq:
            top_words = word_freq.most_common(10)
            words, freqs = zip(*top_words)
            ax9.barh(range(len(words)), freqs, color='lightblue')
            ax9.set_yticks(range(len(words)))
            ax9.set_yticklabels(words)
            ax9.set_xlabel('Frequency')
            ax9.set_title('Top 10 Most Frequent Words')
        
        # 10. Sentiment Box Plot
        ax10 = plt.subplot(4, 3, 10)
        sentiment_data = []
        sentiment_labels = []
        
        for category in self.sentiment_categories.keys():
            cat_scores = [t['polarity'] for t in self.processed_tweets 
                         if t['sentiment_category'] == category]
            if cat_scores:
                sentiment_data.append(cat_scores)
                sentiment_labels.append(category)
        
        if sentiment_data:
            bp = ax10.boxplot(sentiment_data, labels=sentiment_labels, patch_artist=True)
            for patch, label in zip(bp['boxes'], sentiment_labels):
                patch.set_facecolor(self.colors[label])
                patch.set_alpha(0.7)
        
        ax10.set_ylabel('Polarity Score')
        ax10.set_title('Sentiment Category Box Plot')
        ax10.tick_params(axis='x', rotation=45)
        ax10.grid(True, alpha=0.3)
        
        # 11. Cumulative Sentiment Distribution
        ax11 = plt.subplot(4, 3, 11)
        sorted_scores = sorted(self.sentiment_scores)
        y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax11.plot(sorted_scores, y_vals, linewidth=2)
        ax11.set_xlabel('Polarity Score')
        ax11.set_ylabel('Cumulative Probability')
        ax11.set_title('Cumulative Distribution Function')
        ax11.grid(True, alpha=0.3)
        ax11.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax11.legend()
        
        # 12. Statistical Summary
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')
        
        stats_text = f"""
        Statistical Summary:
        
        Total Tweets: {len(self.processed_tweets)}
        Mean Polarity: {np.mean(self.sentiment_scores):.4f}
        Std Deviation: {np.std(self.sentiment_scores):.4f}
        Median: {np.median(self.sentiment_scores):.4f}
        Min Score: {min(self.sentiment_scores):.4f}
        Max Score: {max(self.sentiment_scores):.4f}
        
        Skewness: {pd.Series(self.sentiment_scores).skew():.4f}
        Kurtosis: {pd.Series(self.sentiment_scores).kurtosis():.4f}
        
        Positive: {sum(1 for s in self.sentiment_scores if s > 0.1)}/{len(self.sentiment_scores)}
        Negative: {sum(1 for s in self.sentiment_scores if s < -0.1)}/{len(self.sentiment_scores)}
        Neutral: {sum(1 for s in self.sentiment_scores if -0.1 <= s <= 0.1)}/{len(self.sentiment_scores)}
        """
        
        ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def create_wordcloud_analysis(self):
        """Generate word clouds for different sentiment categories"""
        print("Generating word clouds...")
        
        # Separate texts by sentiment
        positive_texts = []
        negative_texts = []
        neutral_texts = []
        
        for tweet in self.processed_tweets:
            if tweet['polarity'] > 0.1:
                positive_texts.append(tweet['clean_text'])
            elif tweet['polarity'] < -0.1:
                negative_texts.append(tweet['clean_text'])
            else:
                neutral_texts.append(tweet['clean_text'])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # All tweets wordcloud
        all_text = ' '.join([tweet['clean_text'] for tweet in self.processed_tweets])
        if all_text.strip():
            wordcloud_all = WordCloud(width=400, height=300, background_color='white',
                                    colormap='viridis', max_words=100).generate(all_text)
            axes[0, 0].imshow(wordcloud_all, interpolation='bilinear')
            axes[0, 0].set_title('All Tweets Word Cloud')
            axes[0, 0].axis('off')
        
        # Positive tweets wordcloud
        if positive_texts:
            positive_text = ' '.join(positive_texts)
            wordcloud_pos = WordCloud(width=400, height=300, background_color='white',
                                    colormap='Greens', max_words=100).generate(positive_text)
            axes[0, 1].imshow(wordcloud_pos, interpolation='bilinear')
            axes[0, 1].set_title('Positive Tweets Word Cloud')
            axes[0, 1].axis('off')
        
        # Negative tweets wordcloud
        if negative_texts:
            negative_text = ' '.join(negative_texts)
            wordcloud_neg = WordCloud(width=400, height=300, background_color='white',
                                    colormap='Reds', max_words=100).generate(negative_text)
            axes[1, 0].imshow(wordcloud_neg, interpolation='bilinear')
            axes[1, 0].set_title('Negative Tweets Word Cloud')
            axes[1, 0].axis('off')
        
        # Neutral tweets wordcloud
        if neutral_texts:
            neutral_text = ' '.join(neutral_texts)
            wordcloud_neu = WordCloud(width=400, height=300, background_color='white',
                                    colormap='Greys', max_words=100).generate(neutral_text)
            axes[1, 1].imshow(wordcloud_neu, interpolation='bilinear')
            axes[1, 1].set_title('Neutral Tweets Word Cloud')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def perform_clustering_analysis(self):
        """Perform text clustering to find similar tweet groups"""
        print("Performing clustering analysis...")
        
        if len(self.processed_tweets) < 10:
            print("Not enough tweets for clustering analysis.")
            return
        
        # Prepare texts for vectorization
        texts = [tweet['clean_text'] for tweet in self.processed_tweets]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', 
                                   ngram_range=(1, 2), min_df=2)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Determine optimal number of clusters
        max_clusters = min(10, len(self.processed_tweets) // 3)
        if max_clusters < 2:
            print("Not enough tweets for meaningful clustering.")
            return
            
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
            silhouette_avg = silhouette_score(tfidf_matrix.toarray(), cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Choose optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
        
        # Add cluster labels to tweets
        for i, tweet in enumerate(self.processed_tweets):
            tweet['cluster'] = cluster_labels[i]
        
        # Analyze clusters
        print(f"\nClustering Results (Optimal k = {optimal_k}):")
        print("-" * 50)
        
        for cluster_id in range(optimal_k):
            cluster_tweets = [tweet for tweet in self.processed_tweets 
                            if tweet['cluster'] == cluster_id]
            cluster_sentiments = [tweet['polarity'] for tweet in cluster_tweets]
            
            print(f"\nCluster {cluster_id + 1}:")
            print(f"  Size: {len(cluster_tweets)} tweets")
            print(f"  Avg Sentiment: {np.mean(cluster_sentiments):.3f}")
            print(f"  Std Sentiment: {np.std(cluster_sentiments):.3f}")
            print(f"  Sample tweets:")
            for i, tweet in enumerate(cluster_tweets[:3]):
                print(f"    {i+1}. {tweet['clean_text'][:100]}...")
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(tfidf_matrix.toarray())
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Tweet Clusters Visualization (PCA)')
        plt.colorbar(scatter)
        
        # Add cluster centers
        centers_2d = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def export_results(self, filename='sentiment_analysis_results.json'):
        """Export analysis results to JSON file"""
        results = {
            'summary': {
                'total_tweets': len(self.processed_tweets),
                'avg_polarity': np.mean(self.sentiment_scores),
                'std_polarity': np.std(self.sentiment_scores),
                'avg_subjectivity': np.mean([t['subjectivity'] for t in self.processed_tweets])
            },
            'sentiment_distribution': dict(Counter([t['sentiment_category'] for t in self.processed_tweets])),
            'top_positive_tweets': sorted([
                {'text': t['text'], 'polarity': t['polarity'], 'category': t['sentiment_category']}
                for t in self.processed_tweets
            ], key=lambda x: x['polarity'], reverse=True)[:10],
            'top_negative_tweets': sorted([
                {'text': t['text'], 'polarity': t['polarity'], 'category': t['sentiment_category']}
                for t in self.processed_tweets
            ], key=lambda x: x['polarity'])[:10],
            'detailed_tweets': [
                {
                    'id': t['id'],
                    'text': t['text'],
                    'clean_text': t['clean_text'],
                    'polarity': t['polarity'],
                    'subjectivity': t['subjectivity'],
                    'sentiment_category': t['sentiment_category'],
                    'features': t['features'],
                    'cluster': t.get('cluster', None)
                }
                for t in self.processed_tweets
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {filename}")

    def generate_interactive_plotly_dashboard(self):
        """Create interactive Plotly dashboard"""
        print("Generating interactive dashboard...")
        
        if not self.processed_tweets:
            print("No data to visualize.")
            return
        
        # Prepare data
        df = pd.DataFrame([
            {
                'polarity': t['polarity'],
                'subjectivity': t['subjectivity'],
                'sentiment_category': t['sentiment_category'],
                'text_length': t['features']['length'],
                'word_count': t['features']['word_count'],
                'text': t['text'][:100] + '...' if len(t['text']) > 100 else t['text'],
                'timestamp': t['timestamp']
            }
            for t in self.processed_tweets
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Sentiment Distribution', 'Polarity vs Subjectivity',
                          'Sentiment Over Time', 'Text Length vs Sentiment',
                          'Word Count Distribution', 'Sentiment Categories'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "pie"}]]
        )
        
        # 1. Sentiment Distribution Bar Chart
        sentiment_counts = df['sentiment_category'].value_counts()
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                   name="Sentiment Distribution",
                   marker_color=[self.colors[cat] for cat in sentiment_counts.index]),
            row=1, col=1
        )
        
        # 2. Polarity vs Subjectivity Scatter
        fig.add_trace(
            go.Scatter(x=df['polarity'], y=df['subjectivity'],
                      mode='markers',
                      marker=dict(color=df['polarity'], colorscale='RdYlBu', 
                                showscale=True, size=8),
                      text=df['text'],
                      name="Polarity vs Subjectivity"),
            row=1, col=2
        )
        
        # 3. Sentiment Over Time (if timestamp data available)
        try:
            df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
            daily_sentiment = df.groupby(df['date'].dt.date)['polarity'].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=daily_sentiment['date'], y=daily_sentiment['polarity'],
                          mode='lines+markers',
                          name="Daily Average Sentiment"),
                row=2, col=1
            )
        except:
            # If timestamp parsing fails, create a simple index-based plot
            fig.add_trace(
                go.Scatter(x=list(range(len(df))), y=df['polarity'],
                          mode='lines',
                          name="Sentiment Trend"),
                row=2, col=1
            )
        
        # 4. Text Length vs Sentiment
        fig.add_trace(
            go.Scatter(x=df['text_length'], y=df['polarity'],
                      mode='markers',
                      marker=dict(color=df['polarity'], colorscale='Viridis', size=6),
                      text=df['text'],
                      name="Length vs Sentiment"),
            row=2, col=2
        )
        
        # 5. Word Count Distribution
        fig.add_trace(
            go.Histogram(x=df['word_count'], nbinsx=20,
                        name="Word Count Distribution",
                        marker_color='lightblue'),
            row=3, col=1
        )
        
        # 6. Sentiment Categories Pie Chart
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                   marker_colors=[self.colors[cat] for cat in sentiment_counts.index],
                   name="Sentiment Categories"),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Interactive Sentiment Analysis Dashboard",
            showlegend=False
        )
        
        fig.show()

    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Comprehensive Sentiment Analysis...")
        print("=" * 60)
        
        # Get user input
        keyword = input("Enter keyword/tag to search for (or press Enter for all tweets): ").strip()
        if not keyword:
            keyword = None
        
        try:
            num_tweets_input = input("Enter number of tweets to analyze (or press Enter for all): ").strip()
            num_tweets = int(num_tweets_input) if num_tweets_input else None
        except ValueError:
            print("Invalid number. Using all matching tweets.")
            num_tweets = None
        
        # Load and analyze data
        total_loaded = self.load_and_preprocess_data()
        if total_loaded == 0:
            print("No data found in the CSV file.")
            return
        
        self.analyze_sentiments(keyword, num_tweets)
        
        if not self.processed_tweets:
            print("No tweets to analyze after filtering.")
            return
        
        # Generate all analyses
        self.generate_basic_statistics()
        
        print("\nGenerating visualizations...")
        self.create_advanced_visualizations()
        
        self.create_wordcloud_analysis()
        
        if len(self.processed_tweets) >= 10:
            self.perform_clustering_analysis()
        
        # Export results
        export_choice = input("\nExport results to JSON? (y/n): ").strip().lower()
        if export_choice == 'y':
            filename = input("Enter filename (or press Enter for default): ").strip()
            if not filename:
                filename = 'sentiment_analysis_results.json'
            self.export_results(filename)
        
        # Interactive dashboard
        dashboard_choice = input("Generate interactive dashboard? (y/n): ").strip().lower()
        if dashboard_choice == 'y':
            self.generate_interactive_plotly_dashboard()
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

    def analyze_temporal_patterns(self):
        """Analyze sentiment patterns over time periods"""
        if not self.processed_tweets:
            return
        
        print("\nTemporal Pattern Analysis:")
        print("-" * 40)
        
        # Group by different time periods
        hourly_sentiment = defaultdict(list)
        daily_sentiment = defaultdict(list)
        
        for tweet in self.processed_tweets:
            try:
                # Extract hour and day from timestamp
                timestamp = tweet['timestamp']
                if len(timestamp) >= 10:
                    date_part = timestamp[:10]
                    daily_sentiment[date_part].append(tweet['polarity'])
                    
                    if len(timestamp) >= 13:
                        hour = timestamp[11:13]
                        hourly_sentiment[hour].append(tweet['polarity'])
            except:
                continue
        
        # Analyze hourly patterns
        if hourly_sentiment:
            print("Hourly Sentiment Patterns:")
            for hour in sorted(hourly_sentiment.keys()):
                avg_sentiment = np.mean(hourly_sentiment[hour])
                print(f"  {hour}:00 - Avg: {avg_sentiment:.3f} ({len(hourly_sentiment[hour])} tweets)")
        
        # Analyze daily patterns
        if daily_sentiment:
            print(f"\nDaily Sentiment Patterns:")
            for date in sorted(daily_sentiment.keys())[-7:]:  # Last 7 days
                avg_sentiment = np.mean(daily_sentiment[date])
                print(f"  {date} - Avg: {avg_sentiment:.3f} ({len(daily_sentiment[date])} tweets)")

    def detect_sentiment_outliers(self):
        """Detect and analyze sentiment outliers"""
        if not self.processed_tweets:
            return
        
        print("\nSentiment Outlier Detection:")
        print("-" * 40)
        
        scores = np.array(self.sentiment_scores)
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = []
        for i, score in enumerate(scores):
            if score < lower_bound or score > upper_bound:
                outliers.append((i, score, self.processed_tweets[i]['text']))
        
        print(f"Found {len(outliers)} outliers:")
        for i, (idx, score, text) in enumerate(outliers[:10]):  # Show top 10
            text_preview = text[:80] + "..." if len(text) > 80 else text
            print(f"  {i+1}. Score: {score:.3f} | {text_preview}")

def main():
    """Main function to run the sentiment analyzer"""
    try:
        # Get CSV file path
        csv_path = input("Enter the path to your CSV file (default: 'twitter_dataset.csv'): ").strip()
        if not csv_path:
            csv_path = "twitter_dataset.csv"
        
        # Initialize analyzer
        analyzer = AdvancedSentimentAnalyzer(csv_path)
        
        # Run comprehensive analysis
        analyzer.run_comprehensive_analysis()
        
        # Additional analyses
        analyzer.analyze_temporal_patterns()
        analyzer.detect_sentiment_outliers()
        
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main()
