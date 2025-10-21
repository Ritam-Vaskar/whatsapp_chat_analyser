# 💬 WhatsApp Chat Analyzer - Advanced AI-Powered Insights Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49.1-FF4B4B?style=for-the-badge&logo=streamlit)
![Transformers](https://img.shields.io/badge/Transformers-4.53.3-orange?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An advanced AI-powered analytics platform that transforms WhatsApp chat exports into comprehensive psychological, behavioral, and conversational insights using state-of-the-art NLP and machine learning techniques.**

[Features](#-key-features) • [Tech Stack](#-technology-stack) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-technical-documentation) • [Deployment](#-deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Feature Documentation](#-comprehensive-feature-documentation)
- [Machine Learning Components](#-machine-learning-components)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**WhatsApp Chat Analyzer** is a sophisticated, production-ready analytics platform designed to extract deep insights from WhatsApp chat exports. Leveraging cutting-edge Natural Language Processing (NLP), transformer models, and advanced statistical analysis, the platform provides:

- **Emotional Intelligence Analysis** using RoBERTa transformer models
- **Topic Discovery** with BERTopic and sentence embeddings
- **Behavioral Pattern Recognition** through advanced feature engineering
- **Predictive Modeling** using Markov Chain text generation
- **Multi-dimensional Personality Profiling** based on typing patterns
- **Relationship Health Metrics** with custom scoring algorithms

The platform processes chat data through 8 distinct analytical pipelines, each employing specialized ML algorithms and NLP techniques to generate actionable insights.

---

## ✨ Key Features

### 1. 😊 **Emotional Timeline Analysis**
- **Technology**: Cardiff NLP's Twitter-RoBERTa emotion classifier
- **Capabilities**:
  - Real-time emotion detection across 7 categories (joy, sadness, anger, fear, surprise, disgust, neutral)
  - Batch processing with configurable batch sizes for performance optimization
  - Temporal emotion tracking with hourly aggregation
  - Multi-user emotional journey visualization with Plotly
  - Color-coded emotion mapping for intuitive understanding

### 2. 🔥 **Activity Pattern Heatmaps**
- **Technology**: NumPy arrays with statistical aggregation
- **Capabilities**:
  - Hourly activity patterns (24-hour cycle)
  - Day-of-week activity distribution (7-day cycle)
  - Monthly activity trends (12-month cycle)
  - Peak activity detection algorithms
  - Night owl vs. early bird scoring (circadian rhythm analysis)
  - Weekend vs. weekday activity ratios
  - Comparative multi-user activity visualization

### 3. 🎭 **Conversational Style & Personality Profiling**
- **Technology**: Feature engineering with regex-based pattern extraction
- **Capabilities**:
  - 10+ typing style metrics extraction
  - Emoji density analysis
  - Punctuation personality detection
  - Capitalization pattern recognition
  - Abbreviation usage scoring
  - Message length distribution analysis
  - One-word reply frequency
  - Personality archetype classification
  - Custom style tags generation

### 4. 🎯 **Topic Discovery & Evolution**
- **Technology**: BERTopic with MiniLM sentence transformers
- **Capabilities**:
  - Unsupervised topic modeling
  - Automatic optimal topic count determination
  - Conversation thread detection (2-hour gap threshold)
  - Topic keyword extraction (top 5 per topic)
  - Temporal topic evolution tracking
  - User-specific topic preference analysis
  - Monthly topic distribution visualization
  - Dominant topic identification

### 5. 💝 **Friendship Health Score**
- **Technology**: Custom multi-metric scoring algorithm
- **Capabilities**:
  - Response time analysis (median-based)
  - Question balance scoring
  - Conversation initiation equity measurement
  - Emoji usage symmetry analysis
  - Weighted composite friendship score (0-100)
  - Sub-metric breakdown:
    - Reply Speed (30% weight)
    - Question Balance (25% weight)
    - Initiation Balance (25% weight)
    - Emoji Balance (20% weight)
  - Relationship health categorization

### 6. 🔮 **Contextual Chat Prediction**
- **Technology**: 2nd-order Markov Chain text generation
- **Capabilities**:
  - User-specific language model building
  - Probabilistic next-word prediction
  - Context-aware message generation
  - Conversation starter generation
  - Multi-exchange conversation simulation
  - Keyword-seeded predictions
  - Statistical word frequency analysis
  - Personalized response generation

### 7. 😀 **Emoji Sentiment Analysis**
- **Technology**: Custom emoji sentiment mapping (100+ emojis)
- **Capabilities**:
  - Comprehensive emoji sentiment scoring (-1.0 to +1.0)
  - 5-category sentiment classification
  - User emoji personality profiling
  - Top emoji identification per user
  - Emoji diversity metrics
  - Temporal emoji sentiment tracking
  - Radar chart visualization for sentiment distribution
  - Fun fact generation
  - Mood categorization

### 8. ⌨️ **Advanced Typing Style Analysis**
- **Technology**: Multi-dimensional pattern recognition with regex
- **Capabilities**:
  - 30+ typing pattern metrics
  - Punctuation usage profiling
  - Capitalization style detection
  - Repetition habit analysis
  - Formatting preference extraction
  - Vocabulary richness calculation
  - Slang usage quantification
  - Advanced personality DNA generation
  - Layman-friendly summary reports

---

## 🛠️ Technology Stack

### **Core Framework**
- **Streamlit 1.49.1** - Interactive web application framework
- **Python 3.11** - Primary programming language

### **Machine Learning & NLP**
| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Transformers | Hugging Face Transformers | 4.53.3 | Pre-trained model deployment |
| Emotion Classification | CardiffNLP RoBERTa | Latest | Emotion detection |
| Topic Modeling | BERTopic | 0.17.3 | Unsupervised topic discovery |
| Sentence Embeddings | Sentence-Transformers | 5.1.0 | Text vectorization |
| Embedding Model | all-MiniLM-L6-v2 | Latest | Semantic similarity |
| Deep Learning | PyTorch | 2.7.1 | Model inference |
| Vision (optional) | TorchVision | 0.22.1 | Image processing |

### **Data Processing & Analysis**
- **Pandas 2.2.3** - Structured data manipulation
- **NumPy 2.2.6** - Numerical computations
- **Scikit-learn 1.7.2** - Statistical algorithms
- **SciPy 1.16.1** - Scientific computing

### **Visualization**
- **Plotly 6.3.0** - Interactive charts and graphs
- **Matplotlib 3.10.3** - Static visualizations
- **Seaborn 0.13.2** - Statistical data visualization

### **Text Processing**
- **NLTK 3.9.1** - Natural language toolkit
- **Emoji 2.14.1** - Emoji extraction and analysis
- **Regex 2024.11.6** - Pattern matching
- **LangDetect 1.0.9** - Language detection

### **Performance & Optimization**
- **Accelerate 1.9.0** - Model acceleration
- **NumBA 0.61.2** - JIT compilation
- **ThreadPoolCTL 3.6.0** - Thread management

### **Supporting Libraries**
- **Requests 2.32.3** - HTTP client
- **Beautiful Soup 4.13.4** - HTML/XML parsing
- **PyArrow 21.0.0** - Columnar data format
- **TensorBoard** (via PyTorch) - Model monitoring

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                  │
│                  (User Interaction Layer)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              WhatsAppAnalyzer (Orchestrator)                │
│           Coordinates 8 Analysis Pipelines                  │
└─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬────────────┘
      │     │     │     │     │     │     │     │
┌─────▼─────▼─────▼─────▼─────▼─────▼─────▼─────▼────────────┐
│                   Feature Modules                           │
├─────────────────────────────────────────────────────────────┤
│ 1. EmotionAnalyzer        (Transformer-based)               │
│ 2. ActivityAnalyzer       (Statistical aggregation)         │
│ 3. StyleDetector          (Pattern recognition)             │
│ 4. TopicAnalyzer          (BERTopic clustering)             │
│ 5. FriendshipAnalyzer     (Multi-metric scoring)            │
│ 6. ChatPredictor          (Markov chains)                   │
│ 7. EmojiSentimentAnalyzer (Sentiment mapping)               │
│ 8. TypingStyleAnalyzer    (Feature engineering)             │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Data Processing Layer                        │
├─────────────────────────────────────────────────────────────┤
│ • Chat Parser (Regex-based)                                 │
│ • DataFrame Transformation (Pandas)                         │
│ • Timestamp Normalization                                   │
│ • User Filtering                                            │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Visualization Layer (Plotly/Matplotlib)        │
└─────────────────────────────────────────────────────────────┘
```

### **Processing Pipeline**

```
WhatsApp Export (.txt)
       │
       ▼
[Parser Module]
  • Regex pattern matching
  • Date/time parsing
  • Multi-format support
       │
       ▼
[Data Cleaning]
  • System message filtering
  • Deduplication
  • Timestamp sorting
       │
       ▼
[Analysis Mode Selection]
  • Quick Mode (20% sample, fast)
  • Deep Mode (100% data, comprehensive)
       │
       ▼
[Parallel Feature Extraction]
  • Emotion classification (GPU-accelerated)
  • Topic modeling (CPU-bound)
  • Statistical aggregation
       │
       ▼
[Results Aggregation]
  • JSON formatting
  • Chart generation
  • Report compilation
       │
       ▼
[Interactive Dashboard]
  • 8 tabbed sections
  • Real-time filtering
  • Export capabilities
```

---

## 📦 Installation

### **Prerequisites**
- Python 3.11 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- 4GB+ RAM (8GB recommended for large chats)
- GPU optional (CUDA support for faster inference)

### **Step 1: Clone Repository**
```bash
git clone https://github.com/Ritam-Vaskar/whatsapp_chat_analyser.git
cd whatsapp_chat_analyser
```

### **Step 2: Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **Step 4: Download NLTK Data (if needed)**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### **Troubleshooting Installation**

**Issue**: PyTorch installation fails
```bash
# For CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: Transformers model download fails
- Check internet connection
- Set HF_HOME environment variable for cache location
- Manually download models from Hugging Face Hub

---

## 🚀 Usage

### **Running Locally**

```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

### **Exporting WhatsApp Chats**

#### **Android:**
1. Open WhatsApp → Select chat
2. Tap ⋮ (three dots) → More → Export chat
3. Choose "Without media"
4. Save file to device

#### **iOS:**
1. Open WhatsApp → Select chat
2. Tap contact/group name
3. Scroll down → Export Chat
4. Choose "Without Media"
5. Save file

### **Using the Application**

1. **Upload Chat File**: Click "Choose your WhatsApp chat file" and select `.txt` file
2. **Select Analysis Mode**:
   - **Quick Mode**: 20% sample, faster processing (~30 seconds)
   - **Deep Mode**: Full analysis, comprehensive results (~2-5 minutes)
3. **Run Analysis**: Click "🚀 Run Complete Analysis"
4. **Explore Results**: Navigate through 8 tabbed sections
5. **Export Data**: Download JSON report at the bottom

### **Sample Chat Format**

```
18/08/2025, 10:38 pm - Alice: Hey! How are you?
18/08/2025, 10:40 pm - Bob: I'm good! Just finished work 😊
18/08/2025, 10:42 pm - Alice: That's great! Want to grab coffee tomorrow?
```

---

## 📚 Comprehensive Feature Documentation

### **1. Emotion Analysis Module**

**Technical Implementation:**
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-emotion",
    return_all_scores=True
)
```

**Processing Workflow:**
1. Text preprocessing (remove special chars)
2. Batch inference (configurable batch size)
3. Score aggregation per emotion
4. Temporal grouping (hourly bins)
5. Visualization generation

**Output Metrics:**
- Emotion labels: joy, sadness, anger, fear, surprise, disgust, neutral
- Confidence scores per prediction
- Temporal emotion trends
- User-specific emotional profiles

**Use Cases:**
- Relationship conflict detection
- Mental health monitoring
- Conversation mood tracking
- User emotional intelligence profiling

---

### **2. Activity Pattern Analysis**

**Algorithm:**
```python
# Hourly Activity Matrix Generation
hourly_activity = df.groupby(df['timestamp'].dt.hour).size()
hourly_matrix = np.zeros((24, 1))
hourly_matrix[:, 0] = hourly_activity.reindex(range(24), fill_value=0).values
```

**Metrics Calculated:**
- **Peak Hour**: Hour with maximum message density
- **Most Active Day**: Day with highest message count
- **Night Owl Score**: % of messages sent 10pm-6am
- **Early Bird Score**: % of messages sent 5am-9am
- **Weekend Activity**: Ratio of weekend to weekday messages

**Insights Generated:**
- Circadian rhythm analysis
- Work-life balance indicators
- Communication preference patterns
- Availability windows

---

### **3. Topic Discovery System**

**BERTopic Configuration:**
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_model = BERTopic(
    embedding_model=sentence_model,
    min_topic_size=10,
    nr_topics="auto"
)
```

**Processing Steps:**
1. **Conversation Threading**: Group messages with <2hr gap
2. **Document Creation**: Concatenate messages per thread
3. **Embedding Generation**: Convert to 384-dim vectors
4. **Dimensionality Reduction**: UMAP projection
5. **Clustering**: HDBSCAN algorithm
6. **Topic Extraction**: c-TF-IDF for representative words

**Output:**
- Discovered topics with keywords
- Topic distribution over time
- User topic preferences
- Conversation theme evolution

---

### **4. Friendship Health Scoring**

**Composite Score Formula:**
```
Friendship Score = (Reply_Speed × 0.30) + 
                   (Question_Balance × 0.25) + 
                   (Initiation_Balance × 0.25) + 
                   (Emoji_Balance × 0.20)
```

**Sub-Metric Calculations:**

**Reply Speed Score:**
```python
avg_reply_minutes = median(reply_times)
reply_score = max(0, 100 - (avg_reply_minutes / 5))
```

**Question Balance Score:**
```python
q_ratio_diff = abs(user1_q_ratio - user2_q_ratio)
question_score = max(0, 100 - (q_ratio_diff * 500))
```

**Interpretation:**
- **90-100**: Exceptional friendship
- **75-89**: Strong relationship
- **60-74**: Healthy connection
- **50-59**: Needs attention
- **<50**: Imbalanced dynamic

---

### **5. Markov Chain Prediction**

**Model Architecture:**
- **Order**: 2 (bigram context)
- **Training**: Per-user language models
- **Inference**: Weighted random sampling

**Implementation:**
```python
# Chain structure
chain = defaultdict(Counter)
# Key: (word_n-1, word_n) → Value: {word_n+1: count}

# Generation
for current_bigram in chain:
    next_words = chain[current_bigram]
    predicted_word = weighted_random_choice(next_words)
```

**Applications:**
- Autocomplete suggestions
- User impersonation detection
- Conversation style mimicry
- Chatbot personality modeling

---

### **6. Emoji Sentiment Mapping**

**Sentiment Scale:**
```python
emoji_sentiment_map = {
    '😍': 0.9,   # Very positive
    '😊': 0.8,   # Positive
    '😐': 0.0,   # Neutral
    '😔': -0.7,  # Negative
    '😭': -0.9   # Very negative
}
```

**Analysis Pipeline:**
1. Emoji extraction using `emoji` library
2. Sentiment score lookup
3. Categorical bucketing (5 categories)
4. Temporal aggregation
5. User comparison
6. Radar chart generation

**Personality Types:**
- **Sunshine Spreader**: >30% very positive emojis
- **Positive Vibes**: >40% positive emojis
- **Emotional Expresser**: >20% negative emojis
- **Emoji Collector**: >50 unique emojis
- **Emoji Minimalist**: <10 unique emojis

---

### **7. Typing Pattern Recognition**

**Feature Vector (30+ dimensions):**
```python
features = {
    'avg_message_length': float,
    'avg_words_per_message': float,
    'emoji_density': float,
    'exclamation_ratio': float,
    'question_ratio': float,
    'caps_ratio': float,
    'abbreviation_score': float,
    'ellipsis_usage': float,
    'repeated_letters': float,
    'one_word_replies': float,
    # ... 20 more features
}
```

**Pattern Categories:**
- **Punctuation**: 11 patterns
- **Capitalization**: 1 pattern
- **Repetition**: 3 patterns
- **Formatting**: 6 patterns
- **Vocabulary**: 3 metrics

**Personality Tags:**
- Emoji Poet 🎭
- Exclaimer ⚡
- The Minimalist 🧘
- Storyteller 📖
- CAPS LOCK ENTHUSIAST 📢
- The Questioner ❓
- Text Speak Master 📱

---

## 🤖 Machine Learning Components

### **Model Inventory**

| Model | Parameters | Task | Accuracy | Inference Time |
|-------|-----------|------|----------|----------------|
| cardiffnlp/twitter-roberta-base-emotion | 125M | Emotion Classification | 85%+ | ~50ms/text |
| all-MiniLM-L6-v2 | 22M | Sentence Embedding | N/A | ~20ms/text |
| BERTopic (HDBSCAN) | N/A | Topic Clustering | Unsupervised | ~2s/100 docs |

### **Computational Complexity**

- **Emotion Analysis**: O(n) with batching, GPU-accelerated
- **Topic Modeling**: O(n log n) for embedding + O(n²) for clustering
- **Activity Analysis**: O(n) aggregation
- **Friendship Scoring**: O(n) single-pass
- **Markov Chains**: O(n) training, O(k) inference
- **Emoji Analysis**: O(n) linear scan
- **Typing Analysis**: O(n×m) where m = pattern count

### **Performance Optimization**

**Quick Mode (20% sampling):**
```python
if mode == "quick" and len(messages_df) > 5000:
    messages_df = messages_df.sample(frac=0.2, random_state=42)
```

**Batch Processing:**
```python
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    results = classifier(batch, truncation=True)
```

**Memory Management:**
- Streaming file reading for large exports
- Generator-based processing
- Explicit garbage collection between modules

---

## 🌐 Deployment

### **Docker Deployment**

**Build Image:**
```bash
docker build -t whatsapp-analyzer .
```

**Run Container:**
```bash
docker run -p 8501:8501 whatsapp-analyzer
```

**Dockerfile Configuration:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port", "$PORT", "--server.address", "0.0.0.0"]
```

### **Cloud Platforms**

#### **Render**
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy

#### **Streamlit Cloud**
1. Push to GitHub
2. Connect at share.streamlit.io
3. Auto-deploy on push

#### **Heroku**
1. Create `Procfile`:
   ```
   web: streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0
   ```
2. Deploy via Git:
   ```bash
   heroku create app-name
   git push heroku main
   ```

#### **AWS EC2**
```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip
pip install -r requirements.txt

# Run with PM2
pm2 start "streamlit run src/app.py --server.port 8501" --name whatsapp-analyzer
```

---

## 📂 Project Structure

```
whatsapp_chat_analyser/
│
├── src/
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit application
│   ├── parsers.py                # WhatsApp chat parser
│   │
│   └── features/
│       ├── __init__.py
│       ├── emotion_analyzer.py   # RoBERTa emotion classification
│       ├── activity_analyzer.py  # Activity pattern detection
│       ├── typing_format.py      # Style detection (StyleDetector)
│       ├── topic_analyzer.py     # BERTopic implementation
│       ├── friendship.py         # Friendship scoring algorithm
│       ├── chat_predictor.py     # Markov chain text generation
│       ├── emoji_analyzer.py     # Emoji sentiment mapping
│       └── typing_analysis.py    # Advanced typing profiling
│
├── notebooks/
│   ├── 01_parsing_and_eda.ipynb  # Data exploration
│   └── 02_typing_and_stats.ipynb # Feature analysis
│
├── tests/                         # Unit tests (to be implemented)
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── Procfile                       # Heroku/Render deployment
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 🔬 Technical Specifications

### **Input Requirements**
- **Format**: Plain text (.txt)
- **Encoding**: UTF-8
- **Pattern**: `DD/MM/YY, HH:MM am/pm - Sender: Message`
- **Size**: Tested up to 50,000 messages
- **Languages**: English (primary), extensible to others

### **Output Formats**
- **Interactive Dashboard**: Streamlit web interface
- **JSON Export**: Complete analysis results
- **Charts**: Plotly interactive visualizations
- **Static Images**: Matplotlib/Seaborn exports

### **Performance Benchmarks**
| Chat Size | Quick Mode | Deep Mode |
|-----------|-----------|-----------|
| 1,000 msgs | ~15s | ~30s |
| 5,000 msgs | ~25s | ~90s |
| 10,000 msgs | ~40s | ~180s |
| 50,000 msgs | ~120s | ~600s |

*Tested on: Intel i7, 16GB RAM, no GPU*

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### **Development Setup**
```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/

# Lint
flake8 src/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** - Pre-trained transformer models
- **Streamlit** - Web application framework
- **Cardiff NLP** - Emotion classification model
- **BERTopic** - Topic modeling library
- **Open Source Community** - Various supporting libraries

---

## 📧 Contact & Support

- **Developer**: Ritam Vaskar
- **GitHub**: [@Ritam-Vaskar](https://github.com/Ritam-Vaskar)
- **Issues**: [GitHub Issues](https://github.com/Ritam-Vaskar/whatsapp_chat_analyser/issues)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Made with ❤️ and 🤖 by Ritam Vaskar**

*Transforming conversations into insights, one message at a time.*

</div> whatsapp_chat_analyser