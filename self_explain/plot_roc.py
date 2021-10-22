import logging
import os
import datetime
import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_roc(Y_true, Y_pred, save_dir=None, key=None, xlim=[0, 1], ylim=[0, 1]):
    """ Compute ROC for binary detection (single ROC for both training and validation
    """
    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)
    thresholds = np.linspace(Y_pred.min(), Y_pred.max(), 100)

    Y_true = Y_true.astype(int)
    positive_indices = np.where(Y_true > 0.5)
    negative_indices = np.where(Y_true <= 0.5)
    positive_len = positive_indices[0].size
    negative_len = negative_indices[0].size
    roc = []
    for threshold in thresholds:
        tp = np.sum(Y_pred[positive_indices]>=threshold) / positive_len
        fp = np.sum(Y_pred[negative_indices]>=threshold) / negative_len
        roc.append((fp, tp))
    roc = np.array(roc, dtype=[("fp", "f"), ("tp", "f")])

    accuracy = ((Y_pred>0.5).astype(int) == Y_true).sum()/Y_true.size
    title = f"{key} ROC (accuracy={100*accuracy:.1f}%)"
    fig = plt.figure(num="hist")
    width = (thresholds[1] - thresholds[0])/2
    plt.bar(thresholds-width/2, roc["fp"], width=width, label="false positives")
    plt.bar(thresholds+width/2, roc["tp"], width=width, label="true positives")
    plt.xlabel("thresholds")
    plt.ylabel("detections above threshold")
    plt.legend()
    plt.grid()
    plt.title(f"{key} true/false positives ")
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "hist.png")
    print(f"saving histogram to {filename}")
    fig.savefig( filename )
    plt.close("hist")

    fig = plt.figure(num="roc")
    plt.plot(roc["fp"],roc["tp"], label=save_dir)
    plt.legend(loc="lower right")
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.grid("on")
    plt.title(title)
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    filename = os.path.join(save_dir, "roc.png")
    print(f"saving ROC to {filename}")
    fig.savefig(filename)
    plt.close("roc")


