import re
from transformers import pipeline
import plotly.express as px

class EmotionAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            return_all_scores=True
        )
    
    def analyze_emotion(self, text):
        clean_text = re.sub(r'[^\w\s]', '', text)
        if len(clean_text.strip()) < 3:
            return "neutral"
        
        results = self.classifier(clean_text)
        return max(results[0], key=lambda x: x['score'])['label']
    
    def batch_analyze(self, messages_df):
        messages_df['emotion'] = messages_df['message'].apply(self.analyze_emotion)
        return messages_df

    def create_emotion_timeline(self, df):
        # Group by hour for timeline
        df['hour'] = df['timestamp'].dt.floor('H')
        emotion_timeline = df.groupby(['hour', 'emotion', 'sender']).size().reset_index(name='count')
        
        # Color mapping for emotions
        emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#FF4500',
            'fear': '#8B008B',
            'surprise': '#FF69B4',
            'disgust': '#556B2F',
            'neutral': '#808080'
        }
        
        fig = px.line(
            emotion_timeline, 
            x='hour', 
            y='count',
            color='emotion',
            facet_col='sender',
            color_discrete_map=emotion_colors,
            title="Emotional Journey Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Message Count",
            height=600
        )
        
        return fig
