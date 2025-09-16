import emoji
import pandas as pd
import numpy as np

class EmojiSentimentAnalyzer:
    def __init__(self):
        # Comprehensive emoji sentiment mapping
        self.emoji_sentiment_map = {
            # Positive emotions
            '😀': 0.8, '😃': 0.8, '😄': 0.9, '😁': 0.7, '😆': 0.8,
            '😅': 0.6, '😂': 0.9, '🤣': 0.9, '😊': 0.8, '😇': 0.9,
            '🙂': 0.6, '🙃': 0.5, '😉': 0.7, '😌': 0.7, '😍': 0.9,
            '🥰': 0.9, '😘': 0.8, '😗': 0.7, '😙': 0.7, '😚': 0.7,
            '😋': 0.7, '😛': 0.6, '😜': 0.6, '🤪': 0.6, '😝': 0.6,
            '🤑': 0.6, '🤗': 0.8, '🤭': 0.5, '🤫': 0.3, '🤔': 0.0,
            '❤️': 0.9, '💕': 0.9, '💖': 0.9, '💗': 0.9, '💙': 0.8,
            '💚': 0.8, '💛': 0.8, '🧡': 0.8, '💜': 0.8, '🖤': 0.2,
            '🤍': 0.7, '💯': 0.8, '👍': 0.7, '👏': 0.8, '🙌': 0.8,
            '👌': 0.6, '🤞': 0.5, '✌️': 0.6, '🤟': 0.7, '🤘': 0.6,
            '👊': 0.5, '✊': 0.5, '🤛': 0.4, '🤜': 0.4, '🤝': 0.7,
            '🙏': 0.6, '✍️': 0.5, '🎉': 0.9, '🎊': 0.9, '🎈': 0.7,
            '🌟': 0.8, '⭐': 0.8, '🔥': 0.6, '💪': 0.7, '🏆': 0.8,
            
            # Negative emotions
            '😟': -0.6, '😞': -0.7, '😔': -0.7, '😕': -0.5, '🙁': -0.6,
            '😣': -0.6, '😖': -0.6, '😫': -0.7, '😩': -0.7, '🥺': -0.4,
            '😢': -0.8, '😭': -0.9, '😤': -0.5, '😠': -0.8, '😡': -0.9,
            '🤬': -0.9, '🤯': -0.3, '😳': -0.2, '🥵': -0.3, '🥶': -0.3,
            '😱': -0.7, '😨': -0.7, '😰': -0.7, '😥': -0.6, '😓': -0.5,
            '🤔': 0.0, '🤨': -0.2, '😐': 0.0, '😑': -0.1, '😶': 0.0,
            '😏': 0.2, '😒': -0.4, '🙄': -0.3, '😬': -0.2, '🤥': -0.4,
            '💔': -0.9, '🖕': -0.8, '👎': -0.6, '💩': -0.5, '🤮': -0.8,
            
            # Neutral/Mixed
            '😪': -0.3, '😴': 0.1, '🤤': 0.1, '😋': 0.6, '🤓': 0.3,
            '🧐': 0.1, '🤠': 0.4, '🥳': 0.8, '🥴': -0.2, '🤢': -0.6,
            '🤧': -0.2, '😷': -0.1, '🤒': -0.4, '🤕': -0.5, '🤑': 0.3
        }
        
        self.emotion_categories = {
            'very_positive': (0.7, 1.0),
            'positive': (0.3, 0.7),
            'neutral': (-0.3, 0.3),
            'negative': (-0.7, -0.3),
            'very_negative': (-1.0, -0.7)
        }
    
    def extract_emoji_sentiment(self, messages_df):
        emoji_data = []
        
        for _, row in messages_df.iterrows():
            message_emojis = emoji.emoji_list(row['message'])
            
            for emoji_info in message_emojis:
                emoji_char = emoji_info['emoji']
                sentiment_score = self.emoji_sentiment_map.get(emoji_char, 0.0)
                
                emoji_data.append({
                    'sender': row['sender'],
                    'timestamp': row['timestamp'],
                    'emoji': emoji_char,
                    'sentiment_score': sentiment_score,
                    'emoji_description': emoji.demojize(emoji_char)
                })
        
        return pd.DataFrame(emoji_data)
    
    def categorize_emoji_usage(self, emoji_df):
        user_emoji_stats = {}
        
        for sender in emoji_df['sender'].unique():
            user_emojis = emoji_df[emoji_df['sender'] == sender]
            
            # Calculate sentiment distribution
            sentiment_counts = {category: 0 for category in self.emotion_categories}
            
            for _, emoji_row in user_emojis.iterrows():
                score = emoji_row['sentiment_score']
                for category, (min_val, max_val) in self.emotion_categories.items():
                    if min_val <= score < max_val:
                        sentiment_counts[category] += 1
                        break
            
            # Calculate percentages
            total_emojis = len(user_emojis)
            sentiment_percentages = {
                category: (count / total_emojis * 100) if total_emojis > 0 else 0
                for category, count in sentiment_counts.items()
            }
            
            # Find most used emojis
            top_emojis = user_emojis['emoji'].value_counts().head(5).to_dict()
            
            # Calculate average sentiment
            avg_sentiment = user_emojis['sentiment_score'].mean() if len(user_emojis) > 0 else 0
            
            user_emoji_stats[sender] = {
                'sentiment_distribution': sentiment_percentages,
                'top_emojis': top_emojis,
                'average_sentiment': avg_sentiment,
                'total_emoji_count': total_emojis,
                'unique_emojis': len(user_emojis['emoji'].unique())
            }
        
        return user_emoji_stats
    
    def create_emoji_radar_chart(self, user_emoji_stats):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(user_emoji_stats), figsize=(6*len(user_emoji_stats), 6), subplot_kw=dict(projection='polar'))
        
        if len(user_emoji_stats) == 1:
            axes = [axes]
        
        categories = list(self.emotion_categories.keys())
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, (user, stats) in enumerate(user_emoji_stats.items()):
            ax = axes[idx]
            
            values = [stats['sentiment_distribution'][cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=user, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, max(max(values), 20))
            ax.set_title(f"{user}'s Emoji Sentiment Radar\nAvg Sentiment: {stats['average_sentiment']:.2f}")
            ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def generate_emoji_insights(self, user_emoji_stats):
        insights = {}
        
        for user, stats in user_emoji_stats.items():
            # Determine dominant emotion
            dominant_emotion = max(stats['sentiment_distribution'], 
                                 key=stats['sentiment_distribution'].get)
            
            # Emoji personality
            personality_traits = []
            
            if stats['sentiment_distribution']['very_positive'] > 30:
                personality_traits.append("Sunshine Spreader ☀️")
            elif stats['sentiment_distribution']['positive'] > 40:
                personality_traits.append("Positive Vibes 🌈")
            
            if stats['sentiment_distribution']['negative'] > 20:
                personality_traits.append("Emotional Expresser 💭")
            
            if stats['unique_emojis'] > 50:
                personality_traits.append("Emoji Collector 🎭")
            elif stats['unique_emojis'] < 10:
                personality_traits.append("Emoji Minimalist 🎯")
            
            # Fun facts
            fun_facts = []
            if stats['total_emoji_count'] > 100:
                fun_facts.append(f"Used {stats['total_emoji_count']} emojis total!")
            
            top_emoji = list(stats['top_emojis'].keys())[0] if stats['top_emojis'] else "None"
            fun_facts.append(f"Favorite emoji: {top_emoji}")
            
            insights[user] = {
                'dominant_emotion': dominant_emotion,
                'personality_traits': personality_traits,
                'fun_facts': fun_facts,
                'emoji_mood': self._get_emoji_mood(stats['average_sentiment'])
            }
        
        return insights
    
    def _get_emoji_mood(self, avg_sentiment):
        if avg_sentiment >= 0.5:
            return "Super Happy 😄"
        elif avg_sentiment >= 0.2:
            return "Generally Positive 😊"
        elif avg_sentiment >= -0.2:
            return "Balanced 😐"
        elif avg_sentiment >= -0.5:
            return "Somewhat Negative 😔"
        else:
            return "Needs Cheering Up 😢"

def create_emoji_timeline(emoji_df):
    # Group by date and sentiment category
    emoji_df['date'] = emoji_df['timestamp'].dt.date
    
    daily_sentiment = emoji_df.groupby(['date', 'sender']).agg({
        'sentiment_score': ['mean', 'count']
    }).reset_index()
    
    daily_sentiment.columns = ['date', 'sender', 'avg_sentiment', 'emoji_count']
    
    # Create timeline visualization
    fig = px.scatter(daily_sentiment, 
                     x='date', 
                     y='avg_sentiment',
                     size='emoji_count',
                     color='sender',
                     title="Emoji Sentiment Journey Over Time",
                     hover_data=['emoji_count'])
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Neutral Line")
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Emoji Sentiment",
        height=600
    )
    
    return fig