import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,
    precision_recall_curve, 
    auc,
    f1_score
)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin

def ordered_train_test_split(X, y, X_embeddings = None, test_size=0.2, random_state=42,shuffle=True, stratify = True, under_sample=False):
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify= y if stratify else None
    )
    train_idx_sorted = np.sort(train_idx)
    val_idx_sorted = np.sort(val_idx)
    
    X_train = X[train_idx_sorted]
    y_train = y[train_idx_sorted]
    X_val = X[val_idx_sorted]
    y_val = y[val_idx_sorted]
    
    if under_sample:
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train== 0)
        num_min = np.minimum(num_neg, num_pos)
        idx_neg_under_sampled = np.random.choice(np.where(y_train==0)[0], num_min, replace=False)
        idx_pos_under_sampled = np.random.choice(np.where(y_train==1)[0], num_min, replace=False)
        idx_under_sampled = np.sort(np.concat([idx_neg_under_sampled, idx_pos_under_sampled]))
        
        X_train = X_train[idx_under_sampled]
        y_train = y_train[idx_under_sampled]
        
    if X_embeddings is not None:
        X_train_embeddings = X_embeddings[train_idx_sorted]
        X_val_embeddings = X_embeddings[val_idx_sorted]
        
        if under_sample:
            X_train_embeddings = X_train_embeddings[idx_under_sampled]
            
        return X_train, X_train_embeddings, y_train, X_val, X_val_embeddings, y_val
    else:
        return X_train,None, y_train, X_val ,None, y_val
    
def plot_train_test_cm(y_train,y_test, y_proba_train, y_proba, title=None):
    fig, axes = plt.subplots(1,2,figsize=(10,7))
    y_pred = (y_proba > 0.5).astype(int)
    y_pred_train = (y_proba_train > 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=axes[0], colorbar=False)
    axes[0].set_title(f"Train F1-score: {f1_score(y_train, y_pred_train):.2f}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1], colorbar=False)
    axes[1].set_title(f"Test F1-score: {f1_score(y_test, y_pred):.2f}")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_roc(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0,1], [0,1], 'k--', label="Random guess")  
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve\nAUC = {auc:.2f}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
def plot_pr(y_true, y_pred_proba):

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")

    # baseline: proportion of positives
    baseline = np.mean(y_true)
    plt.plot([0,1], [baseline, baseline], 'k--', label="Baseline")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve\nAP = {pr_auc:.2f}")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()
    


# Plot ROC and PR curves side by side
def plot_roc_pr(y_true, y_pred_proba, title=None):
    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Compute PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    baseline = np.mean(y_true)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # --- ROC plot (left) ---
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axes[0].plot([0, 1], [0, 1], 'k--', label="Random guess")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve\nAUC={roc_auc:.2f}")
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # --- PR plot (right) ---
    axes[1].plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
    axes[1].plot([0, 1], [baseline, baseline], 'k--', label="Baseline")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve\nAP = {pr_auc:.2f}")
    axes[1].legend(loc="lower left")
    axes[1].grid(True)


    axes[2].hist(y_true, density=True, label="actual", alpha=0.4)
    axes[2].hist(y_pred_proba, density=True, label = "predicted",alpha=0.4)
    axes[2].set_title("Distribution of predicted probabilities")
    axes[2].legend()
    if title:
        fig.suptitle(title)
    plt.tight_layout()


class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
                    [
                        len(text),                      # char length
                        len(text.split()),              # word count
                        sum(c.isupper() for c in text), # uppercase count
                        text.count("!"),                # exclamation marks
                        text.count("?"),                # question marks
                    ]
            for text in X
        ])
        
    def transform(self, X):
        features = []
        for text in X:
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            num_commas = text.count(",")
            num_semicolons = text.count(";")
            num_quotes = text.count('"')
            num_exclaims = text.count("!")
            num_questions = text.count("?")
            num_upper = sum(c.isupper() for c in text)
            je_ratio = words.count("je") / word_count if word_count else 0
            nous_ratio = words.count("nous") / word_count if word_count else 0
            il_est_ratio = text.count("il est") / word_count if word_count else 0
            # append all
            features.append([
                char_count, word_count, avg_word_len, num_upper,
                num_exclaims, num_questions, num_commas, num_semicolons, num_quotes,
                je_ratio, nous_ratio, il_est_ratio
            ])
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(["char_count",
                         "word_count", 
                         "avg_word_len",
                         "num_upper",
                         "num_exclaims",
                         "num_questions",
                         "num_commas",
                         "num_semicolons",
                         "num_quotes",
                         "je_ratio",
                         "nous_ratio",
                         "il_est_ratio"
                         ])