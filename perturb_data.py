import pandas as pd
import numpy as np
import random

save_suffix = [
    "pep_cls_net_cluster_0107_05.csv",
    "pep_cls_net_cluster_wide_0107_04pep.csv",
    "new_setting_c_edges.csv",
]


def reduct_data():
    for _save_suffix in save_suffix:
        df = pd.read_csv("./debug_data/{0}".format(_save_suffix))
        for fold_id in range(1, 6):
            cur_fold = "Fold{0}_split".format(fold_id)
            for ratio_v in range(5, 35, 5):
                edges_df_train = df[df[cur_fold] == "train"]
                edges_df_test = df[df[cur_fold] == "test"]
                reduct_ratio = 0.01 * ratio_v
                # edges_df_train = (
                #     edges_df.groupby("prot_idx")
                #     .apply(lambda x: x.sample(frac=(1 - self.reduct_ratio)))
                #     .reset_index(drop=Z8LfMLIkLq7dL0yTrue)
                # )
                dfs = []
                s_column = "prot_idx"

                ratio = 1 - reduct_ratio
                for _v in list(edges_df_train[s_column].unique()):
                    _s_df = edges_df_train[edges_df_train[s_column] == _v]
                    if len(_s_df) == 1:
                        dfs.append(_s_df)
                    else:
                        n_rows_to_choose = int(len(_s_df) * ratio)
                        if n_rows_to_choose < 1:
                            dfs.append(_s_df.sample(n=1))
                        else:
                            dfs.append(_s_df.sample(n=n_rows_to_choose))
                edges_df_train = pd.concat(dfs)
                edges_df = pd.concat([edges_df_train, edges_df_test])
                edges_df[["pep_idx", "prot_idx", cur_fold]].to_csv(
                    "data/reduct/fold_{}_ratio_{}_{}".format(
                        fold_id, ratio_v, _save_suffix
                    ),
                    index=False,
                )


def sample_non_existing_edges(num_samples, edge_matrix, len_A, len_B):
    sampled_edges = []

    while len(sampled_edges) < num_samples:
        i = random.randint(0, len_A - 1)
        j = random.randint(0, len_B - 1)
        if not edge_matrix[i][j]:
            sampled_edges.append((i, j))

    return np.array((sampled_edges))


def add_data():
    for _save_suffix in save_suffix:
        df = pd.read_csv("./debug_data/{0}".format(_save_suffix))
        prot_vocab = pd.read_csv("./debug_data/prot_vocab.csv")
        pep_vocab = pd.read_csv("./debug_data/pep_vocab.csv")
        len_pep = len(pep_vocab)
        len_prot = len(prot_vocab)

        edge_matrix = np.zeros((len_pep, len_prot))
        # Generate the existing edges set
        for _pep_idx, _prot_idx in zip(df["pep_idx"], df["prot_idx"]):
            edge_matrix[_pep_idx][_prot_idx] = 1

        for fold_id in range(1, 6):
            cur_fold = "Fold{0}_split".format(fold_id)
            for ratio_v in range(5, 35, 5):
                num_samples_to_add = int(len(df) * 0.01 * ratio_v)

                s_df = df[["pep_idx", "prot_idx", cur_fold]]
                neg_edges = sample_non_existing_edges(
                    num_samples_to_add, edge_matrix, len_pep, len_prot
                )
                add_edges = pd.DataFrame.from_dict(
                    {
                        "pep_idx": neg_edges[:, 0].tolist(),
                        "prot_idx": neg_edges[:, 1].tolist(),
                        cur_fold: ["train"] * len(neg_edges),
                    }
                )
                res_df = pd.concat([s_df, add_edges])

                res_df.to_csv(
                    "data/add/fold_{}_ratio_{}_{}".format(
                        fold_id, ratio_v, _save_suffix
                    ),
                    index=False,
                )


def mix_data():
    for _save_suffix in save_suffix:
        df = pd.read_csv("./debug_data/{0}".format(_save_suffix))
        prot_vocab = pd.read_csv("./debug_data/prot_vocab.csv")
        pep_vocab = pd.read_csv("./debug_data/pep_vocab.csv")
        len_pep = len(pep_vocab)
        len_prot = len(prot_vocab)

        edge_matrix = np.zeros((len_pep, len_prot))
        # Generate the existing edges set
        for _pep_idx, _prot_idx in zip(df["pep_idx"], df["prot_idx"]):
            edge_matrix[_pep_idx][_prot_idx] = 1
        for fold_id in range(1, 6):
            cur_fold = "Fold{0}_split".format(fold_id)
            for ratio_v in range(5, 35, 5):
                edges_df_train = df[df[cur_fold] == "train"]
                edges_df_test = df[df[cur_fold] == "test"]
                reduct_ratio = 0.01 * ratio_v
                # edges_df_train = (
                #     edges_df.groupby("prot_idx")
                #     .apply(lambda x: x.sample(frac=(1 - self.reduct_ratio)))
                #     .reset_index(drop=Z8LfMLIkLq7dL0yTrue)
                # )
                dfs = []
                s_column = "prot_idx"

                num_samples_to_add = int(len(df) * 0.01 * ratio_v)

                neg_edges = sample_non_existing_edges(
                    num_samples_to_add, edge_matrix, len_pep, len_prot
                )
                add_edges = pd.DataFrame.from_dict(
                    {
                        "pep_idx": neg_edges[:, 0].tolist(),
                        "prot_idx": neg_edges[:, 1].tolist(),
                        cur_fold: ["train"] * len(neg_edges),
                    }
                )

                ratio = 1 - reduct_ratio
                for _v in list(edges_df_train[s_column].unique()):
                    _s_df = edges_df_train[edges_df_train[s_column] == _v]
                    if len(_s_df) == 1:
                        dfs.append(_s_df)
                    else:
                        n_rows_to_choose = int(len(_s_df) * ratio)
                        if n_rows_to_choose < 1:
                            dfs.append(_s_df.sample(n=1))
                        else:
                            dfs.append(_s_df.sample(n=n_rows_to_choose))
                edges_df_train = pd.concat(dfs)
                edges_df = pd.concat([edges_df_train, edges_df_test])
                s_df = edges_df[["pep_idx", "prot_idx", cur_fold]]
                res_df = pd.concat([s_df, add_edges])

                res_df.to_csv(
                    "data/mix/fold_{}_ratio_{}_{}".format(
                        fold_id, ratio_v, _save_suffix
                    ),
                    index=False,
                )


# reduct_data()
# add_data()
mix_data()
