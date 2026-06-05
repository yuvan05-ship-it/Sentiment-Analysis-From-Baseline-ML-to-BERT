# Sentiment Analysis: HuggingFace BERT vs Traditional ML

## Problem
Given a movie review, predict whether it is **positive** or **negative**.

This project compares two approaches:
- Traditional ML: TF-IDF + Logistic Regression
- Modern AI: Pre-trained HuggingFace BERT pipeline

---

## Dataset
- **Source:** IMDB 50K Movie Reviews (Kaggle)
- **Size:** 50,000 reviews
- **Balance:** 25,000 positive / 25,000 negative
- **Columns:** `review` (text), `sentiment` (positive/negative)

---

## Approach

### Baseline — TF-IDF + Logistic Regression
1. Convert reviews to numbers using TF-IDF (top 5000 words)
2. Train Logistic Regression on 80% of data
3. Evaluate on 20% test set

### Advanced — HuggingFace BERT Pipeline
1. Load pre-trained `distilbert-base-uncased-finetuned-sst-2-english`
2. Run on 500 reviews (same subset for fair comparison)
3. Extract label + confidence score for each review

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression + TF-IDF | 89.00% |
| HuggingFace BERT | 90.40% |

---

## Key Insights

**1. BERT is more data efficient**
BERT achieved 90.4% with zero training on this dataset.
Logistic Regression needed 40,000 training examples to reach 89%.

**2. Long reviews confuse BERT**
Reviews over 1000 characters showed lower confidence scores.
BERT truncates at 512 tokens — losing context from long reviews.

**3. High confidence = high accuracy**
Predictions with score > 0.90 were almost always correct.
Predictions with score < 0.60 were frequently wrong.

**4. Both models balanced across classes**
Neither model was biased toward positive or negative —
F1-score = 0.89 for both classes in Logistic Regression.

---

## Visualizations

### Accuracy Comparison
![Accuracy Comparison](screenshots/accuracy_comparison.png)

### Confidence Score Distribution
![Confidence Distribution](screenshots/confidence_distribution.png)

### Confusion Matrices
![Confusion Matrices](screenshots/confusion_matrices.png)

### Review Length vs Confidence
![Length vs Confidence](screenshots/length_vs_confidence.png)

---

## Stack
- Python
- HuggingFace Transformers
- scikit-learn
- pandas
- matplotlib
- seaborn
- Jupyter Notebook

---

## How to Run

```bash
# Install dependencies
pip install transformers pandas scikit-learn matplotlib seaborn

# Open notebook
jupyter notebook notebook.ipynb
```

---

## What I Learned
- How TF-IDF converts text into numbers for ML models
- How BERT tokenizes and understands context in sentences
- How Softmax converts raw logits into confidence scores
- Importance of evaluation over just getting the model to run
- How pre-trained models reduce the need for large training data

---

## Future Improvements
- Fine-tune BERT on full 50K dataset for higher accuracy
- Build a Streamlit app for live review prediction
- Try RAG-based approach for explainable predictions
- Test on other domains: product reviews, Twitter sentiment
