import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ActivityAnalyzer:
    def __init__(self):
        pass
    
    def generate_activity_heatmap(self, df):
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        
        hourly_activity = df.groupby(['hour']).size()
        daily_activity = df.groupby(['day']).size()
        monthly_activity = df.groupby(['month']).size()
        
        # Convert Series to numpy arrays for plotly imshow
        hourly_matrix = np.zeros((24, 1))  # 24 hours
        hourly_matrix[:, 0] = hourly_activity.reindex(range(24), fill_value=0).values
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_matrix = np.zeros((7, 1))  # 7 days
        daily_matrix[:, 0] = daily_activity.reindex(days, fill_value=0).values
        
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_matrix = np.zeros((12, 1))  # 12 months
        monthly_matrix[:, 0] = monthly_activity.reindex(months, fill_value=0).values
        
        return hourly_matrix, daily_matrix, monthly_matrix
    
    def calculate_activity_insights(self, df):
        insights = {}
        
        for sender in df['sender'].unique():
            user_data = df[df['sender'] == sender]
            insights[sender] = {
                'peak_hour': user_data.groupby(user_data['timestamp'].dt.hour).size().idxmax(),
                'most_active_day': user_data.groupby(user_data['timestamp'].dt.day_name()).size().idxmax(),
                'night_owl_score': len(user_data[user_data['timestamp'].dt.hour >= 22]) / len(user_data),
                'early_bird_score': len(user_data[user_data['timestamp'].dt.hour <= 6]) / len(user_data),
                'weekend_activity': len(user_data[user_data['timestamp'].dt.weekday >= 5]) / len(user_data)
            }
        
        return insights

def create_comparative_heatmap(user1_data, user2_data, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(user1_data.T, annot=True, cmap='Blues', ax=ax1)
    ax1.set_title(f"{user1_data.index[0]}'s Activity")
    
    sns.heatmap(user2_data.T, annot=True, cmap='Reds', ax=ax2)
    ax2.set_title(f"{user2_data.index[0]}'s Activity")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig
