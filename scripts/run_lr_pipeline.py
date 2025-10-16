import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize

from src.preprocess import clean_text, map_sentiment
from src.models import build_tfidf_vectorizer, build_logistic_regression

# ==============================
# 1. Load dataset
# ==============================
df = pd.read_csv("data/all-data.csv", encoding='ISO-8859-1')
sentiment_col = df.columns[0]
text_col = df.columns[1]

df['text_clean'] = df[text_col].fillna("").apply(clean_text)
df['label'] = df[sentiment_col].apply(map_sentiment)

print(f"✅ Loaded {len(df)} samples")
print(df.head())

# ==============================
# 2. Train/test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# ==============================
# 3. Vectorize
# ==============================
vectorizer = build_tfidf_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 4. Handle imbalance
# ==============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

# ==============================
# 5. Train Logistic Regression
# ==============================
model = build_logistic_regression()
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)

# ==============================
# 6. Evaluate
# ==============================
print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred, digits=4, target_names=["Negative","Neutral","Positive"]))
print("Accuracy:", accuracy_score(y_test, y_pred))

roc_auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2]), y_proba, multi_class='ovr')
print("ROC-AUC:", roc_auc)
