import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import re

# Import analyzer classes - FIXED: Using absolute imports
from features import (
    EmotionAnalyzer,
    StyleDetector,
    TopicAnalyzer,
    FriendshipAnalyzer,
    ChatPredictor,
    EmojiSentimentAnalyzer,
    TypingStyleAnalyzer,
    ActivityAnalyzer
)
from features.topic_analyzer import analyze_topic_evolution

class WhatsAppAnalyzer:
    def __init__(self):
        self.emotion_analyzer = EmotionAnalyzer()
        self.style_detector = StyleDetector()
        self.topic_analyzer = TopicAnalyzer()
        self.friendship = FriendshipAnalyzer()
        self.chat_predictor = ChatPredictor()
        self.emoji_analyzer = EmojiSentimentAnalyzer()
        self.typing_analyzer = TypingStyleAnalyzer()
        self.activity_analyzer = ActivityAnalyzer()

    def run_complete_analysis(self, messages_df):
        results = {}

        # Feature 1: Emotional Timeline
        messages_df = self.emotion_analyzer.batch_analyze(messages_df)
        results['emotion_timeline'] = self.emotion_analyzer.create_emotion_timeline(messages_df)

        # Feature 2: Activity Heatmaps
        try:
            hourly, daily, monthly = self.activity_analyzer.generate_activity_heatmap(messages_df)
            results['activity_heatmaps'] = {
                'hourly': hourly,
                'daily': daily,
                'monthly': monthly
            }
            results['activity_insights'] = self.activity_analyzer.calculate_activity_insights(messages_df)
        except Exception as e:
            results['activity_heatmaps'] = {'error': str(e)}
            results['activity_insights'] = f"Error: {str(e)}"

        # Feature 3: Conversational Style
        try:
            style_features = self.style_detector.extract_features(messages_df)
            results['style_personalities'] = self.style_detector.classify_personality(style_features)
        except Exception as e:
            results['style_personalities'] = f"Style analysis error: {str(e)}"

        # Feature 4: Topic Discovery
        try:
            conversations, enhanced_df = self.topic_analyzer.preprocess_messages(messages_df)
            topics, topic_words, topic_info = self.topic_analyzer.discover_topics(conversations)
            results['topics'] = {
                'discovered_topics': topic_words,
                'topic_evolution': analyze_topic_evolution(enhanced_df, topics, topic_words)
            }
        except Exception as e:
            results['topics'] = {'error': str(e)}

        # Feature 5: Friendship Score
        try:
            results['friendship_analysis'] = self.friendship.calculate_friendship_score(messages_df)
        except Exception as e:
            results['friendship_analysis'] = f"Friendship analysis error: {str(e)}"

        # Feature 6: Chat Prediction
        try:
            self.chat_predictor.build_chain(messages_df)
            results['chat_predictions'] = self.chat_predictor.generate_contextual_predictions(messages_df)
        except Exception as e:
            results['chat_predictions'] = f"Chat prediction error: {str(e)}"

        # Feature 7: Emoji Sentiment
        try:
            emoji_df = self.emoji_analyzer.extract_emoji_sentiment(messages_df)
            emoji_stats = self.emoji_analyzer.categorize_emoji_usage(emoji_df)
            results['emoji_analysis'] = {
                'user_stats': emoji_stats,
                'insights': self.emoji_analyzer.generate_emoji_insights(emoji_stats),
                'radar_chart': self.emoji_analyzer.create_emoji_radar_chart(emoji_stats)
            }
        except Exception as e:
            results['emoji_analysis'] = {'error': str(e)}

        # Feature 8: Advanced Typing Analysis
        try:
            detailed_features = self.typing_analyzer.extract_detailed_features(messages_df)
            results['typing_analysis'] = self.typing_analyzer.classify_advanced_typing_personality(detailed_features)
        except Exception as e:
            results['typing_analysis'] = f"Typing analysis error: {str(e)}"

        return results

def create_streamlit_app():
    st.set_page_config(
        page_title="WhatsApp Chat Analyzer", 
        page_icon="💬", 
        layout="wide"
    )
    
    st.title("🔍 Advanced WhatsApp Chat Analyzer")
    st.markdown("Upload your WhatsApp chat export and discover amazing insights!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your WhatsApp chat file", 
        type=['txt'],
        help="Export your WhatsApp chat as a .txt file and upload it here"
    )
    
    if uploaded_file is not None:
        # Parse the uploaded file
        chat_data = parse_whatsapp_chat_from_upload(uploaded_file)
        messages_df = pd.DataFrame(chat_data)
        
        # Filter out system messages
        messages_df = messages_df[messages_df['sender'] != 'System']
        
        if len(messages_df) > 0:
            # Initialize analyzer
            analyzer = WhatsAppAnalyzer()
            
            # Show basic stats
            st.subheader("📈 Basic Chat Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Messages", len(messages_df))
            with col2:
                st.metric("Unique Users", messages_df['sender'].nunique())
            with col3:
                date_range = messages_df['timestamp'].max() - messages_df['timestamp'].min()
                st.metric("Chat Duration", f"{date_range.days} days")
            with col4:
                avg_per_day = len(messages_df) / max(date_range.days, 1)
                st.metric("Messages/Day", f"{avg_per_day:.1f}")
            
            # Run analysis
            if st.button("🚀 Run Complete Analysis", type="primary"):
                with st.spinner("Analyzing your chat... This might take a few minutes!"):
                    results = analyzer.run_complete_analysis(messages_df)
                
                # Create tabs for different analyses
                tabs = st.tabs([
                    "😊 Emotions", "🔥 Activity", "🎭 Personality", 
                    "🎯 Topics", "💝 Friendship", "🔮 Predictions",
                    "😀 Emojis", "⌨️ Typing Style"
                ])
                
                with tabs[0]:  # Emotions
                    st.subheader("Emotional Timeline")
                    st.plotly_chart(results['emotion_timeline'], use_container_width=True)
                
                with tabs[1]:  # Activity
                    st.subheader("Activity Heatmaps")
                    
                    heatmap_type = st.selectbox("Choose heatmap type:", 
                                               ["Hourly", "Daily", "Monthly"])
                    
                    if heatmap_type == "Hourly":
                        fig = px.imshow(results['activity_heatmaps']['hourly'], 
                                       title="Hourly Activity Patterns")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.json(results['activity_insights'])
                
                with tabs[2]:  # Personality
                    st.subheader("Conversational Personalities")
                    
                    if isinstance(results['style_personalities'], dict):
                        for user, personality in results['style_personalities'].items():
                            with st.expander(f"🎭 {user}'s Style Card"):
                                if isinstance(personality, dict):
                                    st.json(personality)
                                else:
                                    st.write(personality)
                    else:
                        st.info(results['style_personalities'])
                
                with tabs[3]:  # Topics
                    st.subheader("Conversation Topics")
                    
                    if isinstance(results['topics'], dict) and 'discovered_topics' in results['topics']:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Discovered Topics:**")
                            for topic_id, words in results['topics']['discovered_topics'].items():
                                if isinstance(words, list):
                                    st.write(f"**Topic {topic_id}:** {', '.join(words)}")
                                else:
                                    st.write(f"**Topic {topic_id}:** {words}")
                    else:
                        st.info(str(results['topics']))
                
                with tabs[4]:  # Friendship
                    st.subheader("Friendship Analysis")
                    
                    friendship_data = results['friendship_analysis']
                    if isinstance(friendship_data, dict):
                        
                        # Show overall score
                        score = friendship_data['overall_score']
                        st.metric(
                            "Friendship Health Score", 
                            f"{score}/100",
                            delta=f"{'Great' if score > 75 else 'Good' if score > 50 else 'Needs Work'}"
                        )
                        
                        # Show sub-scores
                        col1, col2, col3, col4 = st.columns(4)
                        sub_scores = friendship_data['sub_scores']
                        
                        with col1:
                            st.metric("Reply Speed", f"{sub_scores['reply_speed']:.1f}")
                        with col2:
                            st.metric("Question Balance", f"{sub_scores['question_balance']:.1f}")
                        with col3:
                            st.metric("Initiation Balance", f"{sub_scores['initiation_balance']:.1f}")
                        with col4:
                            st.metric("Emoji Balance", f"{sub_scores['emoji_balance']:.1f}")
                    else:
                        st.info(friendship_data)
                
                with tabs[5]:  # Predictions
                    st.subheader("What If... Chat Predictions")
                    
                    if isinstance(results['chat_predictions'], dict):
                        col1, col2 = st.columns(2)
                        
                        users = list(results['chat_predictions'].keys())
                        for i, (user, predictions) in enumerate(results['chat_predictions'].items()):
                            with col1 if i % 2 == 0 else col2:
                                st.write(f"**{user}'s Predicted Messages:**")
                                if isinstance(predictions, dict):
                                    for pred_type, prediction in predictions.items():
                                        st.write(f"*{pred_type}:* {prediction}")
                                else:
                                    st.write(predictions)
                    else:
                        st.info(str(results['chat_predictions']))
                
                with tabs[6]:  # Emojis
                    st.subheader("Emoji Sentiment Analysis")
                    
                    if isinstance(results['emoji_analysis'], dict) and 'error' not in results['emoji_analysis']:
                        # Show radar chart
                        if 'radar_chart' in results['emoji_analysis']:
                            try:
                                st.pyplot(results['emoji_analysis']['radar_chart'])
                            except:
                                st.info("Radar chart could not be displayed")
                        
                        # Show insights
                        if 'insights' in results['emoji_analysis']:
                            insights = results['emoji_analysis']['insights']
                            if isinstance(insights, dict):
                                for user, user_insights in insights.items():
                                    with st.expander(f"😀 {user}'s Emoji Personality"):
                                        if isinstance(user_insights, dict):
                                            if 'emoji_mood' in user_insights:
                                                st.write(f"**Emoji Mood:** {user_insights['emoji_mood']}")
                                            if 'personality_traits' in user_insights:
                                                traits = user_insights['personality_traits']
                                                if isinstance(traits, list):
                                                    st.write(f"**Personality Traits:** {', '.join(traits)}")
                                                else:
                                                    st.write(f"**Personality Traits:** {traits}")
                                            if 'fun_facts' in user_insights:
                                                st.write("**Fun Facts:**")
                                                facts = user_insights['fun_facts']
                                                if isinstance(facts, list):
                                                    for fact in facts:
                                                        st.write(f"• {fact}")
                                                else:
                                                    st.write(f"• {facts}")
                                        else:
                                            st.write(user_insights)
                    else:
                        st.info(str(results['emoji_analysis']))
                
                with tabs[7]:  # Typing Style
                    st.subheader("Advanced Typing Analysis")
                    
                    if isinstance(results['typing_analysis'], dict):
                        # Show detailed cards
                        for user, personality in results['typing_analysis'].items():
                            with st.expander(f"⌨️ {user}'s Complete Typing DNA"):
                                if isinstance(personality, dict) and 'personality_summary' in personality:
                                    st.markdown(personality['personality_summary'])
                                    
                                    st.divider()
                                    st.markdown("**Style Tags:**")
                                    tags_html = ' '.join([f'<span style="background-color: #1c1c1c; padding: 4px 8px; margin: 0 0px; border-radius: 12px;">{tag}</span>' for tag in personality['all_tags']])
                                    st.markdown(f"<div style='margin-top: 0px;'>{tags_html}</div>", unsafe_allow_html=True)
                                else:
                                    st.write(str(personality))
                    else:
                        st.info(str(results['typing_analysis']))
                
                # Download results
                st.subheader("📊 Export Results")
                
                # Create downloadable report
                report_data = {
                    'analysis_date': datetime.now().isoformat(),
                    'chat_stats': {
                        'total_messages': len(messages_df),
                        'users': messages_df['sender'].unique().tolist(),
                        'date_range': f"{messages_df['timestamp'].min()} to {messages_df['timestamp'].max()}"
                    },
                    'results': results
                }
                
                # Convert to JSON for download
                import json
                json_str = json.dumps(report_data, indent=2, default=str)
                b64 = base64.b64encode(json_str.encode()).decode()
                
                st.download_button(
                    label="📥 Download Complete Analysis Report (JSON)",
                    data=json_str,
                    file_name=f"whatsapp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.error("No valid messages found in the uploaded file. Please check the format.")
    
    # Instructions
    with st.expander("📋 How to export your WhatsApp chat"):
        st.markdown("""
        **For Android:**
        1. Open WhatsApp and go to the chat you want to analyze
        2. Tap the three dots (⋮) in the top right
        3. Select "More" → "Export chat"
        4. Choose "Without media"
        5. Share the file and save it to your device
        
        **For iOS:**
        1. Open WhatsApp and go to the chat you want to analyze
        2. Tap the contact/group name at the top
        3. Scroll down and tap "Export Chat"
        4. Choose "Without Media"
        5. Save the file to your device
        """)

def parse_whatsapp_chat_from_upload(uploaded_file):
    """Parse uploaded WhatsApp chat file"""
    content = str(uploaded_file.read(), "utf-8")
    lines = content.split('\n')

    messages = []
    # Pattern to match: 18/08/25, 10:38 pm - Sender: Message
    pattern = r'(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}\s(?:am|pm))\s-\s([^:]+):\s(.+)'

    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Try regular message pattern
        match = re.match(pattern, line, re.IGNORECASE)  # Added case insensitive flag
        if match:
            date, time, sender, message = match.groups()
            try:
                # Handle both 2-digit and 4-digit years
                timestamp = datetime.strptime(f"{date} {time}", "%d/%m/%y %I:%M %p")
                messages.append({
                    'timestamp': timestamp,
                    'sender': sender.strip(),
                    'message': message.strip()
                })
            except ValueError:
                try:
                    # Fallback for different time format
                    timestamp = datetime.strptime(f"{date} {time.upper()}", "%d/%m/%y %I:%M %p")
                    messages.append({
                        'timestamp': timestamp,
                        'sender': sender.strip(),
                        'message': message.strip()
                    })
                except ValueError:
                    continue

    return messages

if __name__ == "__main__":
    create_streamlit_app()