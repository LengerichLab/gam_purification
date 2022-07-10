import numpy as np

def find_bin(val, ar):
    # Find the first bin which has left index geq the query.
    # The query belongs in the bin to the left of this one.
    # Assumes the bins are right-inclusive.
    return np.argmin(ar < val)


def only_finite(ar):
    return ar[np.isfinite(ar)]


def get_feat_vals(ebm_global, feat_id):
    good_vals = only_finite(np.array(ebm_global.data(feat_id)['names'])).tolist()
    return good_vals


def calc_density(use_density, feat_vals1, feat_vals2,
    feat_id1, feat_id2, X_train, laplace=0):
    if use_density:
        density = np.zeros((len(feat_vals1), len(feat_vals2))) + laplace
        for x in X_train:
            idx1 = find_bin(x[feat_id1], feat_vals1)
            idx2 = find_bin(x[feat_id2], feat_vals2)
            density[idx1, idx2] += 1
        return density
    else:
        return None
