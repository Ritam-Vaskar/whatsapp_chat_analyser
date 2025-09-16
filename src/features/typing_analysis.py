import emoji
import re
from collections import Counter
import numpy as np

class StyleDetector:
    def __init__(self):
        self.style_features = {}
    
    def extract_features(self, messages_df):
        features = {}
        
        for sender in messages_df['sender'].unique():
            user_messages = messages_df[messages_df['sender'] == sender]['message']
            
            features[sender] = {
                'avg_message_length': user_messages.str.len().mean(),
                'avg_words_per_message': user_messages.str.split().str.len().mean(),
                'emoji_density': self.calculate_emoji_density(user_messages),
                'exclamation_ratio': self.count_punctuation(user_messages, '!'),
                'question_ratio': self.count_punctuation(user_messages, '?'),
                'caps_ratio': self.calculate_caps_ratio(user_messages),
                'abbreviation_score': self.calculate_abbreviation_score(user_messages),
                'ellipsis_usage': self.count_pattern(user_messages, r'\.{2,}'),
                'repeated_letters': self.count_pattern(user_messages, r'(.)\1{2,}'),
                'one_word_replies': self.count_one_word_replies(user_messages)
            }
        
        return features
    
    def calculate_emoji_density(self, messages):
        total_emojis = sum(len(emoji.emoji_list(msg)) for msg in messages)
        total_chars = sum(len(msg) for msg in messages)
        return total_emojis / total_chars if total_chars > 0 else 0
    
    def count_punctuation(self, messages, punct):
        total_punct = sum(msg.count(punct) for msg in messages)
        return total_punct / len(messages) if len(messages) > 0 else 0
    
    def calculate_caps_ratio(self, messages):
        total_caps = sum(sum(1 for c in msg if c.isupper()) for msg in messages)
        total_letters = sum(sum(1 for c in msg if c.isalpha()) for msg in messages)
        return total_caps / total_letters if total_letters > 0 else 0
    
    def calculate_abbreviation_score(self, messages):
        abbreviations = ['lol', 'lmao', 'brb', 'idk', 'omg', 'btw', 'tbh', 'imo']
        total_abbrev = sum(
            sum(1 for abbrev in abbreviations if abbrev in msg.lower()) 
            for msg in messages
        )
        return total_abbrev / len(messages) if len(messages) > 0 else 0

    def count_pattern(self, messages, pattern):
        regex = re.compile(pattern, re.IGNORECASE)
        return sum(len(regex.findall(msg)) for msg in messages) / len(messages) if len(messages) > 0 else 0

    def count_one_word_replies(self, messages):
        return sum(1 for msg in messages if len(msg.split()) == 1) / len(messages) if len(messages) > 0 else 0

    def classify_personality(self, features):
        personalities = {}
        
        for user, user_features in features.items():
            tags = []
            
            # Emoji usage classification
            if user_features['emoji_density'] > 0.1:
                tags.append("Emoji Poet 🎭")
            
            # Exclamation usage
            if user_features['exclamation_ratio'] > 0.3:
                tags.append("Exclaimer ⚡")
            
            # Message length classification
            if user_features['avg_words_per_message'] < 3:
                tags.append("The Minimalist 🧘")
            elif user_features['avg_words_per_message'] > 15:
                tags.append("Storyteller 📖")
            
            # Caps usage
            if user_features['caps_ratio'] > 0.2:
                tags.append("CAPS LOCK ENTHUSIAST 📢")
            
            # Question usage
            if user_features['question_ratio'] > 0.2:
                tags.append("The Questioner ❓")
            
            # Abbreviation usage
            if user_features['abbreviation_score'] > 0.3:
                tags.append("Text Speak Master 📱")
            
            personalities[user] = {
                'primary_tag': tags[0] if tags else "Balanced Communicator 💬",
                'all_tags': tags,
                'style_score': user_features
            }
        
        return personalities

def generate_style_card(user, personality_data):
    card = f"""
    🎭 {user}'s Typing DNA
    ━━━━━━━━━━━━━━━━━━━━━━
    Primary Style: {personality_data['primary_tag']}
    
    📊 Style Metrics:
    • Avg words per message: {personality_data['style_score']['avg_words_per_message']:.1f}
    • Emoji density: {personality_data['style_score']['emoji_density']:.2%}
    • Exclamation usage: {personality_data['style_score']['exclamation_ratio']:.1f}/msg
    • Question frequency: {personality_data['style_score']['question_ratio']:.1f}/msg
    
    🏷️ Style Tags: {', '.join(personality_data['all_tags'])}
    """
    return card
