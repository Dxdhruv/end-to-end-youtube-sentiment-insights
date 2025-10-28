# 🎥 End-to-End YouTube Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.17.0-purple.svg)](https://mlflow.org/)
[![AWS](https://img.shields.io/badge/AWS-EC2-yellow.svg)](https://aws.amazon.com/)
[![DVC](https://img.shields.io/badge/DVC-3.53.0-red.svg)](https://dvc.org/)

A comprehensive machine learning platform that analyzes YouTube video comments to provide real-time sentiment insights, helping users quickly assess video quality and audience reception through advanced NLP techniques and MLOps practices.

## 🎯 Project Overview

This project addresses the challenge of quickly evaluating YouTube video content quality by analyzing comment sentiment. Built with modern MLOps practices, it provides an end-to-end solution from data preprocessing to production deployment, enabling users to make informed decisions about video content before investing time in watching.

### Key Features

- **Real-time Sentiment Analysis**: Analyze YouTube comments with 82% accuracy using LightGBM
- **Interactive Visualizations**: Generate pie charts, trend graphs, and word clouds
- **Chrome Extension**: Browser extension for seamless YouTube integration
- **RESTful API**: Scalable Flask-based API with comprehensive endpoints
- **MLOps Pipeline**: Automated CI/CD with DVC and MLflow integration
- **Cloud Deployment**: AWS EC2 deployment with load balancing

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chrome        │    │   Flask API     │    │   ML Pipeline   │
│   Extension     │◄──►│   (AWS EC2)     │◄──►│   (DVC + MLflow)│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   YouTube API   │    │   Sentiment     │    │   Reddit        │
│   Integration   │    │   Analysis      │    │   Dataset       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Dataset & Model Performance

- **Training Dataset**: 39,794 Reddit comments with labeled sentiments
- **Sentiment Classes**: Positive (1), Neutral (0), Negative (-1)
- **Final Model**: LightGBM with optimized hyperparameters
- **Accuracy**: 82% (achieved in trial 69 of hyperparameter optimization)
- **Feature Engineering**: TF-IDF with n-grams (1,3) and 1000 max features

## 🛠️ Technical Stack

### Machine Learning
- **LightGBM**: Gradient boosting framework for classification
- **TF-IDF Vectorization**: Text feature extraction with n-grams
- **NLTK**: Natural language processing and preprocessing
- **Scikit-learn**: Model evaluation and validation

### MLOps & Infrastructure
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control and pipeline management
- **AWS EC2**: Cloud deployment and hosting
- **Flask**: RESTful API development
- **Docker**: Containerization (CI/CD pipeline)

### Frontend & Integration
- **Chrome Extension**: Browser integration for YouTube
- **YouTube Data API**: Comment fetching and processing
- **Matplotlib/Seaborn**: Data visualization
- **WordCloud**: Text visualization

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
AWS CLI configured
YouTube Data API key
```

### Local Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/end-to-end-youtube-sentiment-insights.git
cd end-to-end-youtube-sentiment-insights
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

3. **Set up DVC**
```bash
dvc pull  # Download data and models
```

4. **Run the ML pipeline**
```bash
dvc repro  # Execute the complete ML pipeline
```

5. **Start the Flask API**
```bash
cd flask_app
python app.py
```

### Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `yt-chrome-plugin-frontend` folder
4. Update the API URL in `popup.js` to point to your deployed endpoint

## 📈 Usage Examples

### API Endpoints

#### Predict Sentiment
```python
import requests

# Single comment prediction
response = requests.post('http://localhost:5000/predict', 
    json={'comments': ['This video is amazing!', 'Not helpful at all']})
print(response.json())
```

#### Generate Visualizations
```python
# Generate sentiment pie chart
response = requests.post('http://localhost:5000/generate_chart',
    json={'sentiment_counts': {'1': 45, '0': 30, '-1': 25}})

# Generate word cloud
response = requests.post('http://localhost:5000/generate_wordcloud',
    json={'comments': ['comment1', 'comment2', ...]})
```

### Chrome Extension Usage

1. Navigate to any YouTube video
2. Click the extension icon in your browser toolbar
3. View real-time sentiment analysis results including:
   - Comment analysis summary with metrics
   - Sentiment distribution pie chart
   - Sentiment trend over time
   - Word cloud visualization
   - Top comments with sentiment labels

## 🔬 Model Development Process

### Experimentation Pipeline

The project follows a systematic approach to model development:

1. **Data Ingestion** (`notebooks/1_Preprocessing_&_EDA.ipynb`)
   - Exploratory data analysis
   - Data quality assessment
   - Missing value handling

2. **Baseline Model** (`notebooks/2_experiment_1_baseline_model.ipynb`)
   - Simple logistic regression baseline
   - Performance benchmarking

3. **Feature Engineering** (`notebooks/3_experiment_2_bow_tfidf.ipynb`)
   - Bag of Words vs TF-IDF comparison
   - N-gram analysis

4. **Hyperparameter Optimization** (`notebooks/7_experiment_6_lightgbm_detailed_hpt.ipynb`)
   - Optuna-based hyperparameter tuning
   - 69 trials to achieve optimal performance

5. **Model Evaluation** (`notebooks/8_stacking.ipynb`)
   - Ensemble methods exploration
   - Final model selection

### Key Technical Decisions

- **LightGBM Selection**: Chosen for its superior performance in text classification tasks
- **TF-IDF with N-grams**: Captures both word-level and phrase-level sentiment patterns
- **Class Balancing**: Addressed imbalanced dataset using LightGBM's built-in balancing
- **Feature Engineering**: 1000 max features to balance performance and computational efficiency

## 🚀 Deployment & MLOps

### CI/CD Pipeline

The project implements automated deployment using:

- **GitHub Actions**: Automated testing and deployment triggers
- **DVC Pipeline**: Reproducible ML workflows
- **MLflow Tracking**: Experiment logging and model versioning
- **AWS EC2**: Scalable cloud deployment

### Deployment Architecture

```
GitHub Repository
       │
       ▼
GitHub Actions (CI/CD)
       │
       ▼
AWS EC2 Instance
       │
       ├── Flask API (Port 5000)
       ├── MLflow UI (Port 5000)
       └── Load Balancer (ELB)
```

### Model Registry

- **MLflow Model Registry**: Centralized model versioning
- **Model Promotion**: Automated model promotion pipeline
- **A/B Testing**: Framework for model comparison
- **Rollback Capability**: Quick model rollback in case of issues

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 82% |
| Precision (Macro) | 0.81 |
| Recall (Macro) | 0.80 |
| F1-Score (Macro) | 0.80 |
| Training Time | ~2 minutes |
| Inference Time | <100ms per comment |

## 🎯 Business Impact

### Target Users
- **Students**: Evaluate educational content quality before watching
- **General Users**: Assess video quality and audience reception
- **Content Creators**: Understand audience sentiment and engagement
- **Product Reviewers**: Analyze product feedback from video comments

### Use Cases
- **Educational Content**: Quickly assess tutorial quality
- **Product Reviews**: Evaluate product reception through comments
- **Entertainment**: Gauge audience reaction to videos
- **Marketing**: Analyze brand sentiment in video content

## 🔮 Future Enhancements

### Planned Improvements
1. **Enhanced Dataset**: Integrate YouTube-specific training data for better accuracy
2. **Production Hardening**: Implement robust error handling and monitoring
3. **Chrome Extension**: Complete the browser extension development
4. **Real-time Processing**: Implement streaming comment analysis
5. **Multi-language Support**: Extend to non-English comments
6. **Advanced Visualizations**: Interactive dashboards and analytics

### Technical Roadmap
- **Model Optimization**: Explore transformer-based models (BERT, RoBERTa)
- **Scalability**: Implement microservices architecture
- **Monitoring**: Add comprehensive logging and alerting
- **Testing**: Implement comprehensive unit and integration tests

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Bappy** - *Initial work* - [entbappy73@gmail.com](mailto:entbappy73@gmail.com)

## 🙏 Acknowledgments

- Reddit community for providing the training dataset
- YouTube for the Data API
- Open source ML libraries (LightGBM, MLflow, DVC)
- AWS for cloud infrastructure

---

⭐ **Star this repository if you found it helpful!**
