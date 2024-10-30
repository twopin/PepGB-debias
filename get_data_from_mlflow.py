#!/usr/bin/env python
# get experiment record from mlflow

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Set the MLflow tracking URI if you're using a remote server
mlflow.set_tracking_uri("http://192.168.1.237:45001")

# Initialize the MLflow client
client = MlflowClient()

experiments = []

# add bias experiments
# experiments += client.search_experiments(
#     filter_string="name LIKE 'bias_ratio_exp%'"
# )

# add neg sampling ratio experiments
experiments += client.search_experiments(
    filter_string="name LIKE 'bias_ratio_exp_0701_prot%'"
)

# experiments += client.search_experiments(
#     filter_string="name LIKE 'bias_ratio_exp_fix%'"
# )

# bias_ratio_exp_fix_prot
exp_ids = [_exp.experiment_id for _exp in experiments]

runs = client.search_runs(experiment_ids=exp_ids)


run_df = pd.DataFrame([{**run.data.metrics, **run.data.params} for run in runs])

# select several columns from run_df

run_df = run_df[
    [
        "exp_name",
        "split_method",
        "neg_sampling_ratio",
        "reduct_ratio",
        "test_auc_mean",
        "test_auc_std",
        "test_aupr_mean",
        "test_aupr_std",
    ]
]
run_df = run_df.sort_values(
    by=["exp_name", "split_method", "reduct_ratio"],
    ascending=[True, True, True],
)
print(run_df)
run_df.to_csv("exp_prot_reduct.csv", index=False)
