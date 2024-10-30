import mlflow

mlflow.set_tracking_uri("http://192.168.1.237:45001")
for split in ["a", "b", "c"]:
    experiment = mlflow.search_experiments(
        filter_string="name = 'ppi_graph_{}'".format(split)
    )[0]
    exp_id = experiment.experiment_id
    runs = mlflow.search_runs(exp_id)

    for run_id in runs["run_id"].tolist():
        mlflow.delete_run(run_id)
