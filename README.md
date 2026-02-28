# Job-Roles-Prediction

A full-stack machine learning system that classifies job roles based on skills, job descriptions, and certifications extracted from online sources. The project covers the complete data science pipeline, from web scraping and exploratory data analysis to model deployment via a Dockerized FastAPI backend and React frontend.

---

## Table of Contents

- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Application](#application)
- [Docker Deployment](#docker-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)

---

## Project Description

**Job-Roles-Prediction** is a multi-class text classification project that predicts a candidate's most likely job role from their professional profile. Given a set of skills, a job description, and certifications, the system returns the top predicted job title along with confidence scores.

The system supports direct text input as well as CV upload in PDF or TXT format. When a CV is uploaded, skills and certifications are automatically extracted using spaCy before being passed to the classifier.

---

## Project Structure

```
Job-Roles-Prediction/
│
├── .github/
│   └── workflows/
│       └── main.yml                    # CI/CD pipeline (GitHub Actions)
│
├── code/
│   ├── scraping.py                     # Web scraping from RemoteOK API
│   ├── 1_Preprocessing.ipynb           # Data cleaning and label encoding
│   ├── 2_Feature_Engineering.ipynb     # TF-IDF, Count, SVD, statistical features
│   ├── 3_Modeling_GridSearch.ipynb     # Model training and hyperparameter optimization
│   ├── 4_MLflow.ipynb                  # Experiment tracking and model registry
│   └── app.py                          # FastAPI prediction service
│
├── data/
│   ├── csv/
│   │   ├── jobs.csv                    # Main dataset (scraped + curated)
│   │   └── mlflow_comparison.csv       # Model comparison results
│   ├── pkl/
│   │   ├── label_encoder.pkl
│   │   ├── preprocessed_data.pkl
│   │   ├── feature_sets.pkl
│   │   ├── modeling_results_gridsearch.pkl
│   │   └── mlflow_info.pkl
│   └── images/                         # EDA visualizations (see section below)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # Main React UI
│   │   └── main.jsx
│   └── package.json
│
├── mlruns/                             # MLflow tracking data
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset (`data/csv/jobs.csv`) was built from two sources:

- **RemoteOK API** scraped via `code/scraping.py`
- **Manually curated entries** to ensure coverage of 119 distinct job titles

Each record contains four columns:

| Column | Description |
|---|---|
| Job Title | Target label (119 unique classes) |
| Skills | Semicolon-separated technical skills |
| Job Description | Free-text description of the role |
| Certifications | Relevant professional certifications |

**Dataset statistics:**

- Total records: 2,458
- Training set: 1,966 samples (80%)
- Test set: 492 samples (20%)
- No missing values

---

## Exploratory Data Analysis

EDA was performed in `code/1_Preprocessing.ipynb` and `code/2_Feature_Engineering.ipynb`. All visualizations are saved in `data/images/`.

### Images to generate and save in `data/images/`

The following plots are relevant to this project and should be produced by your EDA notebook:

**1. `job_distribution.png`**
Bar chart of the top 20 most frequent job titles. Shows that the dataset is relatively balanced, with Backend Developer and Software Engineer appearing most often (~45 occurrences each).

How to generate:
```python
import matplotlib.pyplot as plt
df['Job Title'].value_counts().head(20).plot(kind='barh', figsize=(10, 8))
plt.title('Top 20 Job Title Distribution')
plt.tight_layout()
plt.savefig('../data/images/job_distribution.png')
```

**2. `skills_frequency.png`**
Horizontal bar chart of the 30 most common individual skills across all records. Useful for understanding which technologies dominate the dataset (Python, Git, Docker, etc.).

How to generate:
```python
from collections import Counter
all_skills = [s for row in df['Skills'] for s in row.split(';')]
skill_counts = Counter(all_skills).most_common(30)
# plot and save
```

**3. `text_length_distribution.png`**
Histogram showing the distribution of combined text length (Skills + Description + Certifications) per record. Helps understand the variance in input length before vectorization.

**4. `wordcloud_skills.png`**
Word cloud generated from all skills in the dataset. Visually highlights dominant technologies.

```python
from wordcloud import WordCloud
text = ' '.join(df['Skills'].str.replace(';', ' '))
wc = WordCloud(width=800, height=400).generate(text)
wc.to_file('../data/images/wordcloud_skills.png')
```

**5. `class_balance.png`**
Bar chart showing the count per job title across all 119 classes, sorted by frequency. Demonstrates that the dataset is not perfectly balanced and explains why weighted F1-score was chosen as the main metric.

**6. `tfidf_variance.png`**
Plot of explained variance ratio from TruncatedSVD (300 components applied to TF-IDF with 10,000 features). Shows that 300 components explain approximately 50% of the variance.

**7. `model_comparison.png`**
Horizontal bar chart comparing the weighted F1-score of all 20 model/feature combinations. Clearly shows Random Forest with combined features as the top performer.

How to generate (after running notebook 3):
```python
results_df.sort_values('f1_weighted').plot(
    kind='barh', x='combination', y='f1_weighted', figsize=(10, 10)
)
plt.savefig('../data/images/model_comparison.png')
```

**8. `confusion_matrix_best_model.png`**
Confusion matrix for the best model (Random Forest + Combined features) evaluated on the test set. Use a subset of the most frequent classes for readability.

**9. `feature_importance_tfidf.png`**
Bar chart of the top 30 most important TF-IDF features as ranked by the Random Forest model. Shows which terms most influence classification.

**10. `project_architecture.png`**
A diagram illustrating the end-to-end pipeline: data collection → preprocessing → feature engineering → model training → MLflow tracking → FastAPI backend → React frontend. This can be created manually with a tool like draw.io and exported as PNG.

---

## Machine Learning Pipeline

### Step 1 - Preprocessing (`1_Preprocessing.ipynb`)

- Load `jobs.csv` (semicolon-separated)
- Remove rows with null target values
- Apply text cleaning: lowercase, remove punctuation, normalize whitespace
- Encode target labels with `LabelEncoder` (119 classes)
- Combine Skills, Job Description, and Certifications into a single `Combined_Text` column
- Split into train (80%) and test (20%) sets
- Save `preprocessed_data.pkl` and `label_encoder.pkl`

### Step 2 - Feature Engineering (`2_Feature_Engineering.ipynb`)

Five feature representations were created:

| Name | Description | Dimensions |
|---|---|---|
| TF-IDF | Unigrams and bigrams, sublinear TF scaling | 5,000 |
| Count Vectorizer | Raw term frequencies | 3,000 |
| TF-IDF + SVD | Dimensionality reduction via TruncatedSVD | 300 |
| Statistical | Text length, word count, average word length, unique word ratio | 5 |
| Combined | TF-IDF concatenated with statistical features | 5,005 |

### Step 3 - Modeling with GridSearchCV (`3_Modeling_GridSearch.ipynb`)

Six classifiers were trained and optimized:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC
- Random Forest
- K-Nearest Neighbors
- Decision Tree

Each model was evaluated across four feature configurations, resulting in 20 valid combinations. GridSearchCV with 3-fold cross-validation was used to optimize hyperparameters. The best model was selected based on weighted F1-score.

**Best model: Random Forest with Combined features**

| Metric | Score |
|---|---|
| Test Accuracy | 76.6% |
| F1-Score (weighted) | 75.6% |
| F1-Score (macro) | 73.5% |
| CV Score (3-fold) | 70.9% |

Best hyperparameters: `n_estimators=200`, `max_depth=30`, `min_samples_split=2`

---

## Experiment Tracking with MLflow

All 20 model runs were logged in MLflow (`code/4_MLflow.ipynb`) with the following information per run:

- Model type and feature configuration
- All hyperparameters
- Evaluation metrics: accuracy, precision, recall, F1 (weighted and macro), training time, prediction time
- The best model was registered in the MLflow Model Registry as `Job_Classification_Best_Model`

To launch the MLflow UI:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

---

## Application

The prediction service is implemented in `code/app.py` using FastAPI.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata and performance metrics |
| POST | `/predict` | Predict from structured input (skills, description, certifications) |
| POST | `/predict-cv` | Predict from uploaded CV file (PDF or TXT) |

For CV upload, the system uses spaCy (`en_core_web_sm`) to extract skills and certifications from the raw text before running the classifier.

---

## Docker Deployment

The project is fully containerized with two separate services.

### Backend

```dockerfile
# Dockerfile.backend
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "code.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend

```dockerfile
# Dockerfile.frontend
FROM node:20-slim
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
```

### Running with Docker Compose

```bash
docker-compose up --build
```

Services:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

---

## CI/CD Pipeline

A GitHub Actions workflow (`.github/workflows/main.yml`) runs automatically on every push or pull request to the `main` branch. It performs three jobs:

1. **lint-python** - Runs `flake8` on the `code/` directory to catch syntax errors
2. **build-docker-backend** - Builds the backend Docker image (depends on lint passing)
3. **build-docker-frontend** - Builds the frontend Docker image (depends on lint passing)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 20 or higher
- Docker and Docker Compose (for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Fatmaayadi/Job-Roles-Prediction.git
cd Job-Roles-Prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Execute the notebooks in order

```bash
jupyter notebook code/1_Preprocessing.ipynb
jupyter notebook code/2_Feature_Engineering.ipynb
jupyter notebook code/3_Modeling_GridSearch.ipynb
jupyter notebook code/4_MLflow.ipynb
```

### Run the scraper

```bash
python code/scraping.py
```

### Start the API server directly

```bash
uvicorn code.app:app --host 0.0.0.0 --port 8000 --reload
```

### Start the frontend

```bash
cd frontend
npm install
npm run dev
```

### Start everything with Docker Compose

```bash
docker-compose up --build
```

---

## Results

After training and evaluating 20 model/feature combinations with GridSearchCV, the top three configurations were:

| Rank | Model | Features | F1 (weighted) | Accuracy |
|---|---|---|---|---|
| 1 | Random Forest | Combined (TF-IDF + Stats) | 75.6% | 76.6% |
| 2 | Random Forest | Count Vectorizer | 75.1% | 76.2% |
| 3 | Logistic Regression | Count Vectorizer | 74.1% | 74.2% |

The Random Forest model consistently outperformed other classifiers across feature configurations. Linear models (Logistic Regression, Linear SVC) were competitive but slower to train on the TF-IDF space. KNN performed poorly on sparse high-dimensional representations.

---

## Challenges

- **Dataset imbalance**: Some job titles had as few as 10 samples. This affected recall for minority classes. Weighted metrics were used to account for this.
- **Text normalization**: French and English text were mixed in job descriptions, complicating tokenization and stop-word removal.
- **Feature space size**: TF-IDF with unigrams and bigrams produced up to 10,000 features, requiring careful tuning of `max_features` and `min_df` to avoid overfitting.
- **spaCy extraction accuracy**: Automated skill and certification extraction from raw CV text is imperfect. Patterns were tuned to reduce false positives.
- **scikit-learn version mismatch**: Pickled models trained with scikit-learn 1.6.1 produce warnings when loaded with a newer version. Pinning the version in `requirements.txt` resolves this.

---

## Future Improvements

- Integrate transformer-based models (BERT, RoBERTa) for richer text representations
- Add SMOTE or class weighting to improve recall on underrepresented job titles
- Expand the dataset with more scraping sources (LinkedIn, Indeed via authorized APIs)
- Build a real-time job recommendation engine based on predicted roles
- Deploy to a cloud platform (AWS, GCP, or Azure) with a public endpoint
- Add user authentication and prediction history to the frontend