import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

def gaussian_smoothing(pred, size):
    assert size % 2 == 1, "size must be odd"
    sigma = size / 6
    k = size // 2
    x = np.arange(-k, k + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    pred_padded = np.pad(pred, pad_width=k, mode="edge")
    smoothed = np.convolve(pred_padded, kernel, mode="valid")
    return np.clip(smoothed, 0,1)

def gaussian_smoothing_logit(pred, size):
    """Logit-space Gaussian smoothing"""
    eps = 1e-6
    pred = np.clip(pred, eps, 1 - eps)  # avoid log(0)
    logits = np.log(pred / (1 - pred))
    smoothed_logits = gaussian_smoothing(logits, size)
    smoothed_probs = 1 / (1 + np.exp(-smoothed_logits))
    return np.clip(smoothed_probs, 0, 1)

def ema_smoothing(pred, alpha=0.9):
    """
    Exponential Moving Average (causal / weighted past)
    pred: np.array of probabilities
    alpha: weight for previous smoothed value
    """
    smoothed = np.empty_like(pred)
    smoothed[0] = pred[0]
    for t in range(1, len(pred)):
        smoothed[t] = alpha * smoothed[t-1] + (1-alpha) * pred[t]
    return np.clip(smoothed, 0, 1)

def persistence_smoothing(pred, lambda_=0.9):
    """
    Persistence / Markov prior smoothing
    pred: np.array of probabilities
    lambda_: weight for previous prediction (speaker continuity)
    """
    smoothed = np.empty_like(pred)
    smoothed[0] = pred[0]
    for t in range(1, len(pred)):
        smoothed[t] = lambda_ * smoothed[t-1] + (1 - lambda_) * pred[t]
    return np.clip(smoothed, 0, 1)


def smooth(pred, method="gaussian", **kwargs):
    pred = np.asarray(pred)
    if method == "gaussian":
        return gaussian_smoothing(pred, kwargs.get("size"))
        
    elif method == "logit_gaussian":
        return gaussian_smoothing_logit(pred, kwargs.get("size"))
        
    elif method == "ema":
        return ema_smoothing(pred, kwargs.get("alpha", 0.9))
            
    elif method == "persistence":
        return persistence_smoothing(pred,kwargs.get("lambda_", 0.9))
            
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

class SmoothLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        l1_ratio = 0,
        solver = 'lbfgs',
        class_weight = None,
        C=1,
        max_iter=100,
        smooth_size=17,
        pred_threshold=0.5,
    ):
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.class_weight = class_weight
        self.C = C
        self.max_iter = max_iter
        
        self.smooth_size = smooth_size
        self.pred_threshold = pred_threshold
        
        self.is_fitted_ = False

    def fit(self, X, y):
        self.model_ = LogisticRegression(
            l1_ratio = self.l1_ratio,
            solver = self.solver,
            class_weight = self.class_weight,
            C = self.C,
            max_iter = self.max_iter
        )
        self.model_.fit(X, y)
        self.coef_ = self.model_.coef_
        self.is_fitted_ = True
        return self

    def predict_raw_proba(self, X):
        proba = self.model_.predict_proba(X)
        return proba
    
        
    def predict_proba(self, X, smooth_size = None, pred_threshold=None):
        if not self.is_fitted_:
            raise ValueError("This SmoothLogisticRegression instance is not fitted yet.")
        
        if smooth_size is None:
            smooth_size = self.smooth_size
        if pred_threshold is None:
            pred_threshold = self.pred_threshold

        if smooth_size == 0 and pred_threshold == 0.5:
            return self.predict_raw_proba(X)
        
        proba = self.predict_raw_proba(X)[:,1]
        if smooth_size != 0:
            proba = smooth(proba, method="gaussian", size = smooth_size)
            
        if pred_threshold != 0.5:
            proba = adjust_proba(proba, pred_threshold)
        proba = proba.reshape(-1,1)
        
        return np.hstack([1-proba,proba])

    
    def predict(self, X):
        return (self.predict_proba(X)[:,1] > 0.5).astype(int)
        
    

def plot_smoothing(y_true,y_proba,y_proba_smoothed, slc=slice(None, 1000)):
    y_pred = (y_proba > 0.5).astype(int)
    y_pred_smoothed = (y_proba_smoothed > 0.5).astype(int)
    # fig, axes = plt.subplots(1,2, figsize=(10,7))
    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0],colorbar=False)
    # axes[0].set_title(f"Before smoothing\nF1-score: {f1_score(y_true, y_pred):.2f}")
    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred_smoothed, ax=axes[1],colorbar=False)
    # axes[1].set_title(f"After smoothing\nF1-score: {f1_score(y_true, y_pred_smoothed):.2f}")
    # plt.tight_layout()
    # plt.show()

    fig, axes = plt.subplots(1,2, figsize=(10,7))

    if y_true is not None:
        axes[0].plot(y_true[slc], "-", label="actual", alpha=0.9)
    axes[0].plot(y_proba[slc],"-",label="predicted",alpha=0.3)
    axes[0].plot(y_proba_smoothed[slc],"-",label="smoothed",alpha=0.9)
    axes[0].axhline(0.5, c="red", alpha=0.2)
    axes[0].set_title("Predicted probabilities")

    if y_true is not None:
        axes[1].plot(y_true[slc], ".", label="actual", alpha=0.7)
    axes[1].plot(y_pred[slc] + 0.015,".",label="predicted",alpha=0.3)
    axes[1].plot(y_pred_smoothed[slc] + 0.03,".",label="smoothed",alpha=0.7)
    axes[1].set_title("After cut-off of 0.5")
    fig.suptitle("Temporal smoothing")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    

def adjust_threshold(model, X, y_true, smooth_size = None):
    """Find optimal threshold for F1 score"""
    assert smooth_size is not None or hasattr(model,"predict_proba")
    if hasattr(model, "predict_proba"):
        y_raw = cross_val_predict(model, X, y_true, cv=5, method = "predict_proba")[:, 1]
    else:
        y_raw = cross_val_predict(model, X, y_true, cv=5, method = "predict").astype(float)
    
    if smooth_size is not None:
        y_raw = gaussian_smoothing(y_raw, smooth_size)

    thresholds = np.linspace(0, 1, 1000)
    f1_scores = []
    for t in thresholds:
        y_pred = (y_raw >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    ix = np.argmax(f1_scores)
    best_threshold = thresholds[ix]

    return best_threshold


def adjust_proba(p, t, eps=1e-8):
    """Adjust proba so that the threshold is at 0.5"""
    p = np.clip(p, eps, 1 - eps)
    
    logit_p = np.log(p / (1 - p))
    logit_t = np.log(t / (1 - t))
    
    logit_shifted = logit_p - logit_t
    p_new = 1 / (1 + np.exp(-logit_shifted))
    
    return p_new

def calibrate_proba(y_proba_test, y_proba_train, y_train, alpha=1.):
    """
    Apply prior probability correction to probabilities obtained from a model
    trained on a (possibly) balanced dataset.
    """
    
    # True prior from original data
    pi_real = np.mean(y_train)          # P(y=1)
    pi_real_neg = 1 - pi_real           # P(y=0)

    # Training prior (assumed balanced if you resampled)
    pi_train = 0.5
    pi_train_neg = 0.5

    def correct(p):
        numerator = (p * pi_real) / pi_train
        denominator = numerator + ((1 - p) * pi_real_neg) / pi_train_neg
        return numerator / denominator

    def soften(p):
        p_corr = correct(p)
        return alpha * p_corr + (1 - alpha) * p  # interpolation

    return soften(y_proba_test), soften(y_proba_train)