import numpy as np

from gam_purification.models.ebm import purify_all

# TODO: This uses ebm_global, not XGB, binning.


def purify_xgb_uniform(
    xgb_mains, xgb_pairs, ebm_global, dataset_name, X_means=None, X_stds=None
):
    return purify_xgb(
        xgb_mains,
        xgb_pairs,
        ebm_global,
        False,
        dataset_name,
        "uniform",
        None,
        X_means,
        X_stds,
    )


def purify_xgb_empirical(
    xgb_mains, xgb_pairs, ebm_global, dataset_name, X_train, X_means=None, X_stds=None
):
    return purify_xgb(
        xgb_mains,
        xgb_pairs,
        ebm_global,
        True,
        dataset_name,
        "empirical",
        X_train,
        X_means,
        X_stds,
        0,
    )


def purify_xgb_laplace(
    xgb_mains,
    xgb_pairs,
    ebm_global,
    dataset_name,
    X_train,
    X_means=None,
    X_stds=None,
    laplace=1,
):
    return purify_xgb(
        xgb_mains,
        xgb_pairs,
        ebm_global,
        True,
        dataset_name,
        "laplace",
        X_train,
        X_means,
        X_stds,
        laplace,
    )


def purify_xgb(
    xgb_mains,
    xgb_pairs,
    ebm_global,
    use_density,
    dataset_name,
    move_name,
    X_train=None,
    X_means=None,
    X_stds=None,
    laplace=0,
):
    # Organize the pairs so that the key is in our order.
    pairs = {}
    for (feat_name1, feat_name2), val in xgb_pairs.items():
        feat_id1 = ebm_global.feature_names.index(feat_name1)
        feat_id2 = ebm_global.feature_names.index(feat_name2)
        if feat_id1 < feat_id2:
            my_key = (feat_id1, feat_id2)
            my_mat = val.copy()
        else:
            my_key = (feat_id2, feat_id1)
            my_mat = val.copy().T
        pairs[my_key] = my_mat

    return purify_all(
        xgb_mains,
        pairs,
        ebm_global,
        use_density,
        dataset_name,
        "xgb",
        move_name,
        X_train,
        X_means,
        X_stds,
        laplace,
    )


def get_mains_and_pairs(xgb_pairs_raw, ebm_global, xgb_feature_names=None):
    xgb_mains = {}
    xgb_pairs = {}
    pairwise = 0
    marginal = 0

    if xgb_feature_names is None:
        xgb_feature_names = [
            "f{}".format(i) for i in range(len(ebm_global.feature_names))
        ]

    feature_mapping = {
        xgb_feature_names[i]: ebm_global.feature_names[i]
        for i in range(len(ebm_global.feature_names))
    }
    for tree in xgb_pairs_raw:
        for entry in tree:
            try:
                [leaf_val, path] = entry
            except ValueError:
                continue
            feats = [list(x.keys())[0] for x in path]
            if len(feats) < 2:
                if len(feats) < 1:
                    continue
                feat1 = feature_mapping[feats[0]]
                marginal += 1
                min_val = np.max(
                    [float(list(x.values())[0][0]) for x in path]
                )  # max of the mins
                max_val = np.min(
                    [float(list(x.values())[0][1]) for x in path]
                )  # min of the maxes

                feat_id = ebm_global.feature_names.index(feature_mapping[feats[0]])
                good_vals = np.array(ebm_global.data(feat_id)["names"])
                good_vals = good_vals[np.isfinite(good_vals)].tolist()

                min_idx = np.argmin(good_vals < min_val)  # inclusive
                max_idx = np.argmin(good_vals < max_val)
                if max_val == np.inf:
                    max_idx = len(good_vals)

                try:
                    xgb_mains[feature_mapping[feats[0]]][min_idx:max_idx] += float(
                        leaf_val
                    )
                except KeyError:
                    xgb_mains[feature_mapping[feats[0]]] = np.zeros((len(good_vals),))
                    xgb_mains[feature_mapping[feats[0]]][min_idx:max_idx] += float(
                        leaf_val
                    )
                continue

            feat1 = feature_mapping[feats[0]]
            feat2 = feature_mapping[feats[1]]
            if feats[0] > feats[1]:
                swapped = True
            else:
                swapped = False

            if len(set(feats)) == 2:
                pairwise += 1

                min_val1 = float(list(path[0].values())[0][0])
                max_val1 = float(list(path[0].values())[0][1])
                min_val2 = float(list(path[1].values())[0][0])
                max_val2 = float(list(path[1].values())[0][1])

                feat_id1 = ebm_global.feature_names.index(feat1)
                feat_id2 = ebm_global.feature_names.index(feat2)
                good_vals1 = np.array(ebm_global.data(feat_id1)["names"])
                good_vals1 = np.array(good_vals1[np.isfinite(good_vals1)].tolist())

                min_idx1 = np.argmin(good_vals1 < min_val1)
                max_idx1 = np.argmin(good_vals1 < max_val1)
                if max_val1 == np.inf:
                    max_idx1 = len(good_vals1)

                # TODO: This is using GA2M binning, not XGBoost binning.
                good_vals2 = np.array(ebm_global.data(feat_id2)["names"])
                good_vals2 = np.array(good_vals2[np.isfinite(good_vals2)].tolist())
                min_idx2 = np.argmin(good_vals2 < min_val2)
                max_idx2 = np.argmin(good_vals2 < max_val2)
                if max_val2 == np.inf:
                    max_idx2 = len(good_vals2)

                try:
                    if not swapped:
                        xgb_pairs[feat1, feat2][
                            min_idx1:max_idx1, min_idx2:max_idx2
                        ] += float(leaf_val)
                    else:
                        xgb_pairs[feat2, feat1][
                            min_idx2:max_idx2, min_idx1:max_idx1
                        ] += float(leaf_val)
                except KeyError:
                    if not swapped:
                        xgb_pairs[feat1, feat2] = np.zeros(
                            (len(good_vals1), len(good_vals2))
                        )
                        xgb_pairs[feat1, feat2][
                            min_idx1:max_idx1, min_idx2:max_idx2
                        ] += float(leaf_val)
                    else:
                        xgb_pairs[feat2, feat1] = np.zeros(
                            (len(good_vals2), len(good_vals1))
                        )
                        xgb_pairs[feat2, feat1][
                            min_idx2:max_idx2, min_idx1:max_idx1
                        ] += float(leaf_val)

            else:
                marginal += 1
                min_val = np.max(
                    [float(list(x.values())[0][0]) for x in path]
                )  # max of the mins
                max_val = np.min(
                    [float(list(x.values())[0][1]) for x in path]
                )  # min of the maxes

                feat_id = ebm_global.feature_names.index(feature_mapping[feats[0]])
                good_vals = np.array(ebm_global.data(feat_id)["names"])
                good_vals = good_vals[np.isfinite(good_vals)].tolist()

                min_idx = np.argmin(good_vals < min_val)  # inclusive
                max_idx = np.argmin(good_vals < max_val)
                if max_val == np.inf:
                    max_idx = len(good_vals)

                try:
                    xgb_mains[feature_mapping[feats[0]]][min_idx:max_idx] += float(
                        leaf_val
                    )
                except KeyError:
                    xgb_mains[feature_mapping[feats[0]]] = np.zeros((len(good_vals),))
                    xgb_mains[feature_mapping[feats[0]]][min_idx:max_idx] += float(
                        leaf_val
                    )
    return xgb_mains, xgb_pairs, pairwise, marginal


# For max depth of 2
def parse_xgb_tree(tree, feat_splits):
    # return list of leaf vals, with splits that led to that val
    cur_line = tree.split("\n")[0]
    try:
        next_lines = "\n".join(tree.split("\n")[1:])
    except IndexError:
        print("No next line")
        return

    try:
        cur_split = cur_line.split("[")[1].split("]")[0]
    except IndexError:  # Depth 1 Leaf Node
        leaf_val = float(cur_line.split("=")[1])
        return [[leaf_val, feat_splits]], ""

    try:
        next_split = next_lines.split("\n")[0].split("[")[1].split("]")[0]
    except IndexError:
        # Leaf Node is next
        try:
            feat = cur_split.split("<")[0]
            val = cur_split.split("<")[1]
            leaf1_split = {feat: [0, val]}
            leaf2_split = {feat: [val, np.inf]}
        except IndexError:
            feat = cur_split.split(">")[0]
            val = cur_split.split(">")[1]
            leaf2_split = {feat: [0, val]}
            leaf1_split = {feat: [val, np.inf]}

        leaf1_val = next_lines.split("\n")[0].split("=")[1]
        leaf1_splits = feat_splits.copy()
        leaf1_splits.extend([leaf1_split])
        leaf2_val = next_lines.split("\n")[1].split("\n")[0].split("=")[1]
        leaf2_splits = feat_splits.copy()
        leaf2_splits.extend([leaf2_split])
        if ",no" not in leaf1_val:
            if ",no" not in leaf2_val:
                return [
                    [leaf1_val, leaf1_splits],
                    [leaf2_val, leaf2_splits],
                ], "\n".join(next_lines.split("\n")[2:])
            else:
                return [[leaf1_val, leaf1_splits]], "\n".join(
                    next_lines.split("\n")[2:]
                )
        elif ",no" not in leaf2_val:
            return [[leaf2_val, leaf2_splits]], "\n".join(next_lines.split("\n")[2:])
        else:
            return [], "\n".join(next_lines.split("\n")[2:])

    try:
        feat = cur_split.split("<")[0]
        val = cur_split.split("<")[1]
        tree1_split = {feat: [0, val]}
        tree2_split = {feat: [val, np.inf]}
    except IndexError:
        feat = cur_split.split(">")[0]
        val = cur_split.split(">")[1]
        tree2_split = {feat: [0, val]}
        tree1_split = {feat: [val, np.inf]}

    tree1_splits, sibling_tree = parse_xgb_tree(next_lines, [tree1_split])
    tree2_splits, _ = parse_xgb_tree(sibling_tree, [tree2_split])

    return tree1_splits, tree2_splits
