# Financial Sentiment Analysis - AI574 Group Project

**Team #11: John Pretz | Manas Sahoo**  
**Instructor:** Prof. Bard  
**Course:** Natural Language Processing (AI574) | Fall 2025

---------------------------------------------------------------------------------------------------------------

## Project Overview

Financial texts—such as news articles, social media posts, and earnings calls—shape public perception and impact stock markets. Manual sentiment analysis is impractical due to the massive volume of data. This project leverages **NLP** to automate sentiment extraction, providing actionable insights for investors.

**Objective:** Compare statistical (TF-IDF + Logistic Regression) and neural (FinBERT) NLP approaches to evaluate their effectiveness in capturing financial sentiment.

-------------------------------------------------------------------------------------------------------------------

## Key Features

- **Sources:** News, social media, corporate reports
- **Statistical NLP:** TF-IDF + Logistic Regression with **SMOTE** handling class imbalance
- **Neural NLP:** FinBERT, transformer-based financial sentiment model
- **EDA & Preprocessing:** Text cleaning, label mapping, word clouds, text length and class distributions
- **Evaluation Metrics:** Accuracy, F1-score, ROC-AUC, confusion matrices
- **Visualizations:** Word clouds, prediction comparison, ROC curves, F1-score bar charts
- **Insights:** Misclassification analysis and sample predictions

-----------------------------------------------------------------------------------------------------------

## Dataset Overview

- **Total samples:** 4,845  
- **Sentiment distribution:** Neutral (2,878), Positive (1,363), Negative (604)  
- **Sources:** Financial Phrase Bank, Kaggle datasets, Hugging Face datasets, earnings call transcripts

----------------------------------------------------------------------------------------------------------

## Folder Structure

```text
group-project/
│
├── data/               # Raw and processed datasets
├── docs/               # Project proposal, related work, slides
├── notebooks/          # Jupyter notebooks
├── scripts/            # Python scripts for pipelines
├── src/                # Python modules (preprocessing, models, FinBERT)
├── presentations/      # Presentations
├── venv/               # Virtual environment (ignored in Git)
├── requirements.txt    # Python dependencies
└── README.md           # This file

-------------------------------------------------------------------------------------

Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/JPretz/AI574-2025-FALL-Project
cd group-project

2. Create and activate virtual environment
 Windows
python -m venv venv
venv\Scripts\activate

 macOS/Linux
python -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

-------------------------------------------------------------------------------------
 Running the Pipeline
 1. Logistic Regression (TF-IDF + LR)
python -m scripts.run_lr_pipeline


 2. FinBERT Predictions (sample)
from src.finbert import run_finbert

sample_texts = [
    "The company reported record profits this quarter.",
    "Market volatility continues to concern investors."
]

predictions = run_finbert(sample_texts)
print(predictions)

-----------------------------------------------------------------------------------
 Example Outputs
 TF-IDF + Logistic Regression:
Accuracy: 73.2%
F1-scores:
  Negative: 0.606
  Neutral: 0.822
  Positive: 0.573

 FinBERT (sample predictions):
"The company reported record profits this quarter." --> Positive
"Market volatility continues to concern investors." --> Neutral

------------------------------------------------------------------------------------
 Evaluation Metrics
 Accuracy

 . Precision / Recall / F1-score

 . ROC-AUC (multi-class)

. Confusion Matrix

. Misclassification analysis

. Sample predictions for interpretability
-------------------------------------------------------------------------------------
 References:

Araci, Dogu. FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. 2019.

Ahmed, Wesam et al. Sentiment Analysis on Twitter Using Machine Learning Techniques and TF-IDF. 2023.

Cheng Yu et al. A Deep Learning Framework Integrating CNN and BiLSTM for Financial Systemic Risk Analysis. 2025.

Full reference list in docs/Project Final.pdf
-----------------------------------------------------------------------------------
MIT License

Copyright (c) 2025 John Pretz & Manas Sahoo

