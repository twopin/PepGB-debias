import pickle

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, precision_recall_curve,
                             roc_auc_score)

from src.model import PPIHetero

state = torch.load(
    "./lightning_logs/version_0/checkpoints/epoch=004-val_auc=98.95.ckpt",
    map_location="cpu",
)
with open("./tmp/data_back.pickle", "rb") as fin:
    test_data = pickle.load(fin)
new_state_dict = {}
for key, value in state["state_dict"].items():
    if key.startswith("model."):
        new_state_dict[key[6:]] = value
    else:
        new_state_dict[key] = value

model = PPIHetero(256, 5283, 3318, test_data.metadata())
model.load_state_dict(new_state_dict)

pred = model(test_data).sigmoid().data.numpy()
gt_label = test_data["pep", "bind", "prot"].edge_label.numpy()
pred_label = pred >= 0.10
pred_label = pred_label.astype(np.int64)
print(roc_auc_score(gt_label, pred))
precision_1, recall_1, threshold_1 = precision_recall_curve(gt_label, pred)
aupr_1 = auc(recall_1, precision_1)
print(aupr_1)
print(classification_report(gt_label, pred_label))
