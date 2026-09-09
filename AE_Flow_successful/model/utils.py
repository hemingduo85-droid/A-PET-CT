import torch
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc


@torch.no_grad()
def plot_distribution(labels, scores, epoch):
    import numpy as np
    import matplotlib.pyplot as plt

    normal_scores = np.array([s for s, l in zip(scores, labels) if l == 0])
    abnormal_scores = np.array([s for s, l in zip(scores, labels) if l == 1])

    plt.figure()
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(abnormal_scores, bins=50, alpha=0.5, label='Abnormal', density=True)
    plt.legend()
    plt.title(f"Anomaly Score Distribution at Epoch {epoch}")
    plt.savefig(f"./picture/distribution_epoch_{epoch}.png")
    plt.close()


def metrics(labels, scores):
    from sklearn.metrics import roc_auc_score, average_precision_score

    return {
        'AUROC': roc_auc_score(labels, scores),
        'AP': average_precision_score(labels, scores)
    }


@torch.no_grad()
def get_test_results(model, beta, threshold, normal_loader, abnormal_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomaly_scores = []
    labels = []
    for i, (img, label) in tqdm(enumerate(normal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            anomaly_scores.append(score)
            labels.append(0)
        # break

    for i, (img, label) in tqdm(enumerate(abnormal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            anomaly_scores.append(score)
            labels.append(1)
        # break

    return metrics(labels, anomaly_scores, threshold)




