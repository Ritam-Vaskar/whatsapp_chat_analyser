import re
import numpy as np
import pandas as pd
from collections import Counter

class TypingStyleAnalyzer:
    def __init__(self):
        self.style_patterns = {
            'punctuation': {
                'exclamation_single': r'[^!]![^!]',
                'exclamation_multiple': r'!{2,}',
                'question_single': r'[^?]\?[^?]',
                'question_multiple': r'\?{2,}',
                'ellipsis': r'\.{2,}',
                'comma_usage': r',',
                'period_usage': r'\.',
                'semicolon': r';',
                'colon': r':',
                'dash': r'-{2,}',
                'parentheses': r'\([^)]*\)'
            },
            'capitalization': {
                'all_caps_words': r'\b[A-Z]{2,}\b',
            },
            'repetition': {
                'repeated_letters': r'(.)\1{2,}',
                'repeated_words': r'\b(\w+)\s+\1\b',
                'repeated_punctuation': r'([!?.])\1+',
            },
            'formatting': {
                'asterisk_emphasis': r'\*[^*]+\*',
                'underscore_emphasis': r'_[^_]+_',
                'quotes': r'"[^"]+"',
                'numbers': r'\b\d+\b',
                'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                'mentions': r'@\w+'
            },
        }

    # ------------------ feature extraction ------------------
    def extract_detailed_features(self, messages_df: pd.DataFrame):
        """
        messages_df must have columns: ['sender','message','timestamp']
        """
        detailed_features = {}

        for sender in messages_df['sender'].unique():
            user_msgs = messages_df[messages_df['sender'] == sender].copy()
            msgs = user_msgs['message'].tolist()
            detailed_features[sender] = {
                'message_stats': self._analyze_message_structure(msgs),
                'punctuation_patterns': self._analyze_punctuation(msgs),
                'capitalization_style': self._analyze_capitalization(msgs),
                'repetition_habits': self._analyze_repetition(msgs),
                'formatting_preferences': self._analyze_formatting(msgs),
                'vocabulary_style': self._analyze_vocabulary(msgs),
            }
        return detailed_features

    def _analyze_message_structure(self, messages):
        lens = [len(m) for m in messages]
        words = [len(m.split()) for m in messages]
        return {
            'avg_message_length': np.mean(lens) if lens else 0,
            'avg_words_per_message': np.mean(words) if words else 0,
            'single_word_ratio': sum(1 for w in words if w == 1) / len(words) if words else 0,
        }

    def _analyze_punctuation(self, messages):
        all_text = ' '.join(messages)
        return {k: len(re.findall(p, all_text)) / max(len(messages), 1)
                for k, p in self.style_patterns['punctuation'].items()}

    def _analyze_capitalization(self, messages):
        caps_ratio, total_letters, total_caps = 0, 0, 0
        for m in messages:
            letters = [c for c in m if c.isalpha()]
            total_letters += len(letters)
            total_caps += sum(1 for c in m if c.isupper())
        if total_letters:
            caps_ratio = total_caps / total_letters
        return {'caps_ratio': caps_ratio}

    def _analyze_repetition(self, messages):
        rep = {}
        for name, pat in self.style_patterns['repetition'].items():
            rep[name] = sum(len(re.findall(pat, m)) for m in messages) / max(len(messages), 1)
        return rep

    def _analyze_formatting(self, messages):
        fmt = {}
        for name, pat in self.style_patterns['formatting'].items():
            fmt[name] = sum(len(re.findall(pat, m)) for m in messages) / max(len(messages), 1)
        return fmt

    def _analyze_vocabulary(self, messages):
        words = re.findall(r'\b\w+\b', ' '.join(messages).lower())
        if not words:
            return {}
        uniq = len(set(words))
        slang = ['lol','lmao','idk','brb','omg']
        slang_count = sum(words.count(s) for s in slang)
        return {
            'vocabulary_richness': uniq / len(words),
            'avg_word_length': np.mean([len(w) for w in words]),
            'slang_ratio': slang_count / len(words)
        }

    # ------------------ layman friendly summary ------------------
    def layman_report(self, detailed_features):
        """
        Produce a human-friendly description for each user.
        """
        reports = {}
        for user, f in detailed_features.items():
            ms = f['message_stats']
            pp = f['punctuation_patterns']
            vc = f['vocabulary_style']
            caps = f['capitalization_style'].get('caps_ratio', 0)

            lines = []
            # sentence length
            if ms['avg_words_per_message'] < 5:
                lines.append("Prefers short, quick messages.")
            elif ms['avg_words_per_message'] > 15:
                lines.append("Writes long and detailed messages.")
            else:
                lines.append("Uses medium-length sentences.")

            # punctuation
            if pp['exclamation_multiple'] > 0.1:
                lines.append("Loves using exclamation marks for excitement!")
            if pp['ellipsis'] > 0.1:
                lines.append("Often uses ellipses (…) for dramatic pauses.")

            # capitalization
            if caps > 0.3:
                lines.append("Occasionally types in ALL CAPS for emphasis.")

            # vocabulary
            if vc.get('slang_ratio', 0) > 0.05:
                lines.append("Comfortable with internet slang like lol or idk.")
            if vc.get('vocabulary_richness', 0) > 0.5:
                lines.append("Has a rich and varied vocabulary.")

            # fallback
            if not lines:
                lines.append("Balanced, straightforward typing style.")
            reports[user] = lines
        return reports

    def classify_advanced_typing_personality(self, detailed_features):
        """
        Classify advanced typing personality based on detailed features.
        """
        personality_profiles = {}

        for user, features in detailed_features.items():
            # Message structure analysis
            avg_words = features['message_stats']['avg_words_per_message']
            msg_length_desc = (
                f"Prefers short messages (avg. {avg_words:.1f} words/message)" if avg_words < 5
                else f"Writes detailed messages (avg. {avg_words:.1f} words/message)" if avg_words > 15
                else f"Uses medium-length messages (avg. {avg_words:.1f} words/message)"
            )

            # Punctuation analysis
            punct_patterns = features['punctuation_patterns']
            ellipsis_per_100 = punct_patterns['ellipsis'] * 100
            exclamation_per_100 = punct_patterns['exclamation_multiple'] * 100
            question_per_100 = punct_patterns['question_multiple'] * 100

            # Vocabulary analysis
            vocab = features['vocabulary_style']
            slang_percentage = vocab.get('slang_ratio', 0) * 100
            vocab_richness = vocab.get('vocabulary_richness', 0) * 100

            # Build personality insights
            insights = [msg_length_desc]

            if ellipsis_per_100 > 10:
                insights.append(f"Loves dramatic pauses (...) - {ellipsis_per_100:.1f} per 100 messages")
            if exclamation_per_100 > 10:
                insights.append(f"Enthusiastic typer (!!) - {exclamation_per_100:.1f} per 100 messages")
            if question_per_100 > 10:
                insights.append(f"Curious mind (??) - {question_per_100:.1f} questions per 100 messages")
            if slang_percentage > 5:
                insights.append(f"Uses internet slang - {slang_percentage:.1f}% of messages")
            if vocab_richness > 50:
                insights.append(f"Rich vocabulary - uses {vocab_richness:.1f}% unique words")

            # Determine primary personality tag
            tags = []
            if ellipsis_per_100 > 10:
                tags.append("Dramatic Typist 🎭")
            if exclamation_per_100 > 10:
                tags.append("Enthusiastic Communicator ⚡")
            if question_per_100 > 10:
                tags.append("Curious Mind 🤔")
            if vocab_richness > 50:
                tags.append("Wordsmith ✍️")
            if not tags:
                tags.append("Balanced Communicator 💬")

            # Format the output in a user-friendly way
            personality_profiles[user] = {
                'primary_tag': tags[0],
                'all_tags': tags,
                'personality_summary': f"{user}'s Typing DNA:\n" + 
                                     "\n".join(f"• {insight}" for insight in insights)
            }

        return personality_profiles
