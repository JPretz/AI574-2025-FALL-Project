from transformers import BertTokenizer, BertForSequenceClassification, pipeline

def load_finbert():
    model_name = "yiyanghkust/finbert-tone"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp
