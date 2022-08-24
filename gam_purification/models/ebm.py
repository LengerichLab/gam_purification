import numpy as np

from gam_purification.purify import purify
from gam_purification.utils import (
    calc_density,
    get_feat_vals,
    plot_interaction,
    merge_arrs,
    add_or_new
)

# TODO: Should we return new bins for mains or just smash them together?
def purify_ebm_uniform(
    ebm_global, dataset_name, X_means=None, X_stds=None, should_transpose=True
):
    return purify_ebm(
        ebm_global,
        False,
        dataset_name,
        "uniform",
        X_train=None,
        X_means=X_means,
        X_stds=X_stds,
        should_transpose=should_transpose,
    )


def purify_ebm_empirical(
    ebm_global, dataset_name, X_train, X_means=None, X_stds=None, should_transpose=True
):
    return purify_ebm(
        ebm_global,
        True,
        dataset_name,
        "empirical",
        X_train=X_train,
        X_means=X_means,
        X_stds=X_stds,
        should_transpose=should_transpose,
    )


def purify_ebm_laplace(
    ebm_global,
    dataset_name,
    X_train,
    X_means=None,
    X_stds=None,
    laplace=1,
    should_transpose=True,
):
    return purify_ebm(
        ebm_global,
        True,
        dataset_name,
        "laplace",
        X_train=X_train,
        X_means=X_means,
        X_stds=X_stds,
        laplace=laplace,
        should_transpose=should_transpose,
    )


def purify_ebm(
    ebm_global,
    use_density,
    dataset_name,
    move_name,
    X_train=None,
    X_means=None,
    X_stds=None,
    laplace=0,
    should_transpose=True,
):
    mains = {}
    pairs, pairs_names = {}, {}

    for i, feat_name in enumerate(ebm_global.feature_names):
        my_data = ebm_global.data(i)
        if my_data["type"] == "univariate":
            feat_id = ebm_global.feature_names.index(feat_name)
            mains[feat_id] = my_data["scores"].copy()
        elif my_data["type"] == "interaction":
            feat_name1 = feat_name.split(" x ")[0]
            feat_name2 = feat_name.split(" x ")[1]
            feat_id1 = ebm_global.feature_names.index(feat_name1)
            feat_id2 = ebm_global.feature_names.index(feat_name2)
            if feat_id1 < feat_id2:
                my_key = (feat_id1, feat_id2)
                pairs[my_key] = my_data["scores"].copy()
                pairs_names[my_key] = (my_data["left_names"], my_data["right_names"])
                if should_transpose:
                    pairs[my_key] = pairs[my_key].T
            else:
                my_key = (feat_id2, feat_id1)
                pairs[my_key] = my_data["scores"].copy().T
                print("{} were transposed".format(my_key))

    return purify_all(
        mains,
        pairs,
        pairs_names,
        ebm_global,
        use_density,
        dataset_name,
        "ebm",
        move_name,
        X_train,
        X_means,
        X_stds,
        laplace,
    )


def purify_bivariate(
    ebm_global,
    feat_id1,
    feat_id2,
    mains_moved,
    pairs_moved,
    raw_pair_mat,
    raw_pair_names,
    use_density,
    laplace,
    X_train=None,
):
    assert feat_id1 < feat_id2
    feat_vals1 = get_feat_vals(ebm_global, feat_id1)
    feat_vals2 = get_feat_vals(ebm_global, feat_id2)
    raw_pair_vals1 = raw_pair_names[0]
    raw_pair_vals2 = raw_pair_names[1]

    density = calc_density(
        use_density,
        raw_pair_vals1[: raw_pair_mat.shape[0]],
        raw_pair_vals2[: raw_pair_mat.shape[1]],
        feat_id1,
        feat_id2,
        X_train,
        laplace,
    )
    intercept, m1, m2, pure_mat, n_iters = purify(
        raw_pair_mat, densities=density, tol=1e-6
    )

    pairs_moved[(feat_id1, feat_id2)] = pure_mat.copy()
    try:
        add_or_new(mains_moved, feat_id1, m1)
    except ValueError:  # Different binning between main/pair.
        mapped_m1 = merge_arrs(feat_vals1, raw_pair_vals1, m1)
        add_or_new(mains_moved, feat_id1, mapped_m1)
    try:
        add_or_new(mains_moved, feat_id2, m2)
    except ValueError:  # Different binning between main/pair.
        mapped_m2 = merge_arrs(feat_vals2, raw_pair_vals2, m2)
        add_or_new(mains_moved, feat_id2, mapped_m2)

    return intercept, mains_moved, pairs_moved, feat_vals1, feat_vals2


def get_id_for_feat_name(name, ebm_global):
    return ebm_global.feature_names.index(name)


def purify_all(
    mains,
    pairs,
    pairs_names,
    ebm_global,
    use_density,
    dataset_name,
    model_name,
    move_name,
    X_train=None,
    X_means=None,
    X_stds=None,
    laplace=0,
    should_plot=False,
):
    mains_moved = {}
    pairs_moved = {}
    intercept = 0.0

    for feat_id, vals in mains.items():
        mains_moved[feat_id] = vals.copy()

    for (feat_id1, feat_id2), vals in pairs.items():
        assert feat_id1 < feat_id2

        inter, mains_moved, pairs_moved, feat_vals1, feat_vals2 = purify_bivariate(
            ebm_global,
            feat_id1,
            feat_id2,
            mains_moved,
            pairs_moved,
            vals.copy(),
            pairs_names[(feat_id1, feat_id2)],
            use_density,
            laplace,
            X_train,
        )
        intercept += inter

        if should_plot:
            feat_name1 = ebm_global.feature_names[feat_id1]
            feat_name2 = ebm_global.feature_names[feat_id2]
            if X_stds is not None and X_means is not None:
                plot_interaction(
                    np.array(feat_vals1) * X_stds[feat_id1] + X_means[feat_id1],
                    np.array(feat_vals2) * X_stds[feat_id2] + X_means[feat_id2],
                    vals,
                    pairs_moved[(feat_id1, feat_id2)],
                    dataset_name,
                    feat_name1,
                    feat_name2,
                    model_name,
                    move_name,
                )
            else:
                plot_interaction(
                    feat_vals1,
                    feat_vals2,
                    vals,
                    pairs_moved[(feat_id1, feat_id2)],
                    dataset_name,
                    feat_name1,
                    feat_name2,
                    model_name,
                    move_name,
                )

    return {
        "mains": mains,
        "mains_moved": mains_moved,
        "pairs": pairs,
        "pairs_moved": pairs_moved,
        "intercept": intercept,
    }
