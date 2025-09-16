import random
from collections import defaultdict, Counter

class ChatPredictor:
    def __init__(self, order=2):
        self.order = order
        self.chains = {}
        
    def build_chain(self, messages_df):
        for sender in messages_df['sender'].unique():
            user_messages = messages_df[messages_df['sender'] == sender]['message'].tolist()
            
            # Create chain for this user
            chain = defaultdict(Counter)
            
            for message in user_messages:
                words = message.lower().split()
                if len(words) < self.order + 1:
                    continue
                    
                for i in range(len(words) - self.order):
                    key = tuple(words[i:i + self.order])
                    next_word = words[i + self.order]
                    chain[key][next_word] += 1
            
            self.chains[sender] = chain
    
    def generate_message(self, sender, seed_words=None, max_length=20):
        if sender not in self.chains or not self.chains[sender]:
            return f"{sender} might say: 'Hey there!'"
        
        chain = self.chains[sender]
        
        # Start with seed or random key
        if seed_words and len(seed_words) >= self.order:
            current = tuple(seed_words[-self.order:])
        else:
            current = random.choice(list(chain.keys()))
        
        words = list(current)
        
        for _ in range(max_length - self.order):
            if current not in chain:
                break
                
            # Weighted random selection
            possible_words = chain[current]
            total_count = sum(possible_words.values())
            
            if total_count == 0:
                break
                
            rand_num = random.randint(1, total_count)
            cumulative = 0
            
            for word, count in possible_words.items():
                cumulative += count
                if rand_num <= cumulative:
                    words.append(word)
                    current = tuple(words[-self.order:])
                    break
        
        generated_text = ' '.join(words)
        return f"{sender} might say: '{generated_text}'"
    
    def generate_conversation_prediction(self, num_exchanges=5):
        users = list(self.chains.keys())
        if len(users) < 2:
            return "Need at least 2 users for conversation prediction"
        
        conversation = []
        current_speaker = random.choice(users)
        
        for i in range(num_exchanges):
            message = self.generate_message(current_speaker)
            conversation.append(message)
            
            # Switch speaker
            current_speaker = random.choice([u for u in users if u != current_speaker])
        
        return conversation

    def generate_contextual_predictions(self, messages_df):
        predictions = {}
        
        # Analyze recent conversation context
        recent_messages = messages_df.tail(10)
        recent_topics = self._extract_keywords(recent_messages['message'].tolist())
        
        # Generate themed predictions
        for sender in messages_df['sender'].unique():
            predictions[sender] = {
                'random_message': self.generate_message(sender),
                'contextual_message': self.generate_message(
                    sender, 
                    seed_words=recent_topics[:2] if recent_topics else None
                ),
                'conversation_starter': self.generate_message(
                    sender,
                    max_length=10
                )
            }
        
        return predictions

    def _extract_keywords(self, messages):
        from collections import Counter
        import re
        
        # Simple keyword extraction
        all_words = []
        for message in messages:
            words = re.findall(r'\b\w+\b', message.lower())
            all_words.extend([w for w in words if len(w) > 3])
        
        return [word for word, count in Counter(all_words).most_common(10)]
