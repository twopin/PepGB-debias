import pickle
import os
from jsonargparse import CLI


def extract_feats(src_feat: str = "", save_prefix: str = ""):
    save_dir = os.path.dirname(src_feat)
    with open(src_feat, "rb") as fin:
        orig_feat = pickle.load(fin)

    mean_feat = {k: v["esm_avg"] for k, v in orig_feat.items()}
    max_feat = {k: v["esm_max"] for k, v in orig_feat.items()}
    # save mean feat
    with open(
        os.path.join(save_dir, "{}_mean.pickle".format(save_prefix)), "wb"
    ) as fout:
        pickle.dump(mean_feat, fout)

    # save max feat
    with open(
        os.path.join(save_dir, "{}_max.pickle".format(save_prefix)), "wb"
    ) as fout:
        pickle.dump(max_feat, fout)


if __name__ == "__main__":
    CLI(extract_feats)
