from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_tfidf_vectorizer(max_features=30000):
    return TfidfVectorizer(ngram_range=(1,2), max_features=max_features, stop_words='english')

def build_logistic_regression():
    return LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
