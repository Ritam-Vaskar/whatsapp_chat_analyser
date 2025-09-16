import pandas as pd
import emoji
from datetime import timedelta

class FriendshipAnalyzer:
    def __init__(self):
        self.metrics = {}
    
    def filter_real_users(self, messages_df):
        """Return DataFrame with only real participants."""
        return messages_df
    
    def calculate_reply_speed(self, messages_df):
        messages_df = self.filter_real_users(messages_df)
        reply_times = []
        
        for i in range(1, len(messages_df)):
            current_msg = messages_df.iloc[i]
            prev_msg = messages_df.iloc[i-1]
            
            # Check if it's a reply (different sender)
            if current_msg['sender'] != prev_msg['sender']:
                time_diff = current_msg['timestamp'] - prev_msg['timestamp']
                if time_diff < timedelta(hours=4):  # consider replies within 4 hrs
                    reply_times.append({
                        'replier': current_msg['sender'],
                        'reply_time_minutes': time_diff.total_seconds() / 60
                    })
        
        reply_df = pd.DataFrame(reply_times)
        avg_reply_times = reply_df.groupby('replier')['reply_time_minutes'].median()
        
        return avg_reply_times
    
    def calculate_question_balance(self, messages_df):
        messages_df = self.filter_real_users(messages_df)
        question_stats = {}
        
        for sender in messages_df['sender'].unique():
            user_messages = messages_df[messages_df['sender'] == sender]['message']
            
            questions_asked = sum(1 for msg in user_messages if '?' in msg)
            total_messages = len(user_messages)
            
            question_stats[sender] = {
                'questions_asked': questions_asked,
                'question_ratio': questions_asked / total_messages if total_messages > 0 else 0
            }
        
        return question_stats
    
    def calculate_initiation_ratio(self, messages_df):
        messages_df = self.filter_real_users(messages_df)
        messages_df = messages_df.sort_values('timestamp')
        messages_df['time_gap'] = messages_df['timestamp'].diff()
        
        initiations = messages_df[messages_df['time_gap'] > timedelta(hours=2)]
        initiation_counts = initiations['sender'].value_counts()
        
        return initiation_counts
    
    def calculate_emoji_balance(self, messages_df):
        messages_df = self.filter_real_users(messages_df)
        emoji_stats = {}
        
        for sender in messages_df['sender'].unique():
            user_messages = messages_df[messages_df['sender'] == sender]['message']
            
            total_emojis = sum(len(emoji.emoji_list(msg)) for msg in user_messages)
            total_messages = len(user_messages)
            
            emoji_stats[sender] = {
                'emoji_count': total_emojis,
                'emoji_per_message': total_emojis / total_messages if total_messages > 0 else 0
            }
        
        return emoji_stats
    
    def calculate_friendship_score(self, messages_df):
        messages_df = self.filter_real_users(messages_df)
        users = messages_df['sender'].unique()
        if len(users) != 2:
            return "Friendship score only available for two-person chats"
        
        user1, user2 = users
        
        # Compute metrics
        reply_times = self.calculate_reply_speed(messages_df)
        question_stats = self.calculate_question_balance(messages_df)
        initiation_counts = self.calculate_initiation_ratio(messages_df)
        emoji_stats = self.calculate_emoji_balance(messages_df)
        
        # Reply Speed Score
        avg_reply = reply_times.mean() if len(reply_times) > 0 else 60
        reply_score = max(0, 100 - (avg_reply / 5))  # fast replies ~ high score
        
        # Question Balance Score
        q_ratio_diff = abs(
            question_stats[user1]['question_ratio'] - 
            question_stats[user2]['question_ratio']
        )
        question_balance_score = max(0, 100 - (q_ratio_diff * 500))
        
        # Initiation Balance Score
        total_initiations = initiation_counts.sum()
        if total_initiations > 0:
            initiation_balance = 1 - abs(
                initiation_counts.get(user1, 0) - initiation_counts.get(user2, 0)
            ) / total_initiations
            initiation_score = initiation_balance * 100
        else:
            initiation_score = 50
        
        # Emoji Balance Score
        emoji_diff = abs(
            emoji_stats[user1]['emoji_per_message'] - 
            emoji_stats[user2]['emoji_per_message']
        )
        emoji_balance_score = max(0, 100 - (emoji_diff * 100))
        
        # Overall Friendship Score (weighted)
        friendship_score = (
            reply_score * 0.3 +
            question_balance_score * 0.25 +
            initiation_score * 0.25 +
            emoji_balance_score * 0.2
        )
        
        return {
            'overall_score': round(friendship_score, 1),
            'sub_scores': {
                'reply_speed': round(reply_score, 1),
                'question_balance': round(question_balance_score, 1),
                'initiation_balance': round(initiation_score, 1),
                'emoji_balance': round(emoji_balance_score, 1)
            },
            'raw_metrics': {
                'avg_reply_time_minutes': avg_reply,
                'question_stats': question_stats,
                'initiation_counts': initiation_counts.to_dict(),
                'emoji_stats': emoji_stats
            }
        }
        
import matplotlib.pyplot as plt
import numpy as np

def create_friendship_dashboard(friendship_data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall Score Gauge (pie-style)
    score = friendship_data['overall_score']
    colors = ['red' if score < 50 else 'yellow' if score < 75 else 'green']
    
    ax1.pie(
        [score, 100 - score],
        colors=colors + ['lightgray'],
        startangle=90
    )
    ax1.set_title(f"Friendship Health Score\n{score}/100")
    
    # Sub-scores Radar Chart
    categories = list(friendship_data['sub_scores'].keys())
    values = list(friendship_data['sub_scores'].values())
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]   # close the radar loop
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2)
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 100)
    ax2.set_title("Friendship Dimensions")
    
    # Initiation Balance
    init_data = friendship_data['raw_metrics']['initiation_counts']
    ax3.bar(init_data.keys(), init_data.values(), color="skyblue")
    ax3.set_title("Conversation Initiations")
    ax3.set_ylabel("Count")
    
    # Reply Time Distribution
    reply_time = friendship_data['raw_metrics']['avg_reply_time_minutes']
    ax4.bar(['Average Reply Time'], [reply_time], color="salmon")
    ax4.set_title("Response Time")
    ax4.set_ylabel("Minutes")
    
    plt.tight_layout()
    return fig
