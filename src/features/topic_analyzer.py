from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px

class TopicAnalyzer:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            min_topic_size=10,
            nr_topics="auto"
        )
    
    def preprocess_messages(self, messages_df):
        # Combine messages into conversation threads (gap >2 hrs = new conversation)
        messages_df['conversation_group'] = (
            messages_df['timestamp'].diff() > pd.Timedelta(hours=2)
        ).cumsum()
        
        # Create documents for BERTopic
        conversations = messages_df.groupby('conversation_group')['message'].apply(
            lambda x: ' '.join(x)
        ).tolist()
        
        return conversations, messages_df
    
    def discover_topics(self, conversations):
        if not conversations:
            raise ValueError("No valid conversations found for topic discovery.")

        topics, probabilities = self.topic_model.fit_transform(conversations)
        
        topic_info = self.topic_model.get_topic_info()
        topic_words = {}
        
        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # Skip outlier topic
                words = self.topic_model.get_topic(topic_id)
                topic_words[topic_id] = [word for word, _ in words[:5]]
        
        return topics, topic_words, topic_info


def analyze_topic_evolution(messages_df, topics, topic_words):
    # Add topic assignments back to messages
    conversation_groups = messages_df['conversation_group'].unique()
    if len(conversation_groups) != len(topics):
        raise ValueError("Mismatch between conversation groups and topics")

    # Map topics to conversation groups
    topic_mapping = dict(zip(conversation_groups, topics))
    messages_df['topic'] = messages_df['conversation_group'].map(topic_mapping)

    # Analyze topic distribution over time
    monthly_topics = messages_df.groupby([
        messages_df['timestamp'].dt.to_period('M'), 
        'topic'
    ]).size().reset_index(name='count')

    # Create topic labels
    topic_labels = {
        tid: f"Topic {tid}: {', '.join(words[:3])}"
        for tid, words in topic_words.items()
    }

    monthly_topics['topic_label'] = monthly_topics['topic'].map(topic_labels)

    return monthly_topics, topic_labels


def visualize_topic_evolution(monthly_topics):
    fig = px.bar(
        monthly_topics, 
        x='timestamp', 
        y='count',
        color='topic_label',
        title="Topic Evolution Over Time"
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Messages",
        height=600
    )
    
    return fig


def generate_topic_insights(messages_df, topics, topic_words):
    insights = {
        'dominant_topics': {},
        'user_topic_preferences': {},
        'topic_timeline': {}
    }
    
    # Find dominant topics overall
    topic_counts = pd.Series(topics).value_counts()
    insights['dominant_topics'] = {
        f"Topic {tid}": {
            'keywords': topic_words.get(tid, []),
            'message_count': count,
            'percentage': count / len(topics) * 100
        }
        for tid, count in topic_counts.head(5).items()
        if tid != -1
    }
    
    # User-specific topic preferences
    for sender in messages_df['sender'].unique():
        user_topics = messages_df[messages_df['sender'] == sender]['topic']
        user_topic_dist = user_topics.value_counts(normalize=True)
        
        insights['user_topic_preferences'][sender] = {
            f"Topic {tid}": {
                'percentage': percentage * 100,
                'keywords': topic_words.get(tid, [])
            }
            for tid, percentage in user_topic_dist.head(3).items()
            if tid != -1
        }
    
    return insights
