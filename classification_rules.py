import torch


def _get_categorical_values(X, features_types, feature_idx):
    """
    Internal helper to convert a specific one-hot encoded feature back to its
    categorical integer value using argmax.
    """
    start = sum(features_types[:feature_idx])
    dim = features_types[feature_idx]
    return torch.argmax(X[:, start:start + dim], dim=1)


def upper_half_rule(X, features_types, deciding_feature=0, **kwargs):
    """
    Returns 1.0 if the categorical value of the deciding feature is in the
    lower half of its range (0 to threshold-1), and 0.0 otherwise.
    """
    cat_vals = _get_categorical_values(X, features_types, deciding_feature)
    threshold = features_types[deciding_feature] // 2
    return (cat_vals < threshold).float().unsqueeze(1)


def lower_half_rule(X, features_types, deciding_feature=0, **kwargs):
    """
    Returns 1.0 if the categorical value of the deciding feature is in the
    upper half of its range (threshold to dim-1), and 0.0 otherwise.
    """
    cat_vals = _get_categorical_values(X, features_types, deciding_feature)
    threshold = features_types[deciding_feature] // 2
    return (cat_vals >= threshold).float().unsqueeze(1)


def parity_rule(X, features_types, deciding_feature=0, **kwargs):
    """
    Returns 1.0 if the categorical value of the deciding feature is even,
    and 0.0 if it is odd.
    """
    cat_vals = _get_categorical_values(X, features_types, deciding_feature)
    return (cat_vals % 2 == 0).float().unsqueeze(1)


def probabilistic_shortcut(X, features_types, primary_idx=0, secondary_idx=1, p_agree=1.0, p_disagree=0.8, **kwargs):
    """
    A probabilistic rule comparing two features.
    1. Extracts signals (0 or 1) from primary and secondary features based on
       whether they are in their respective upper halves.
    2. If signals agree: label matches the signal with probability p_agree.
    3. If signals disagree: label matches primary signal with probability p_disagree.
    """

    sig1 = upper_half_rule(X, features_types, primary_idx).squeeze()
    sig2 = upper_half_rule(X, features_types, secondary_idx).squeeze()

    y = torch.zeros(X.shape[0], device=X.device)
    rand_vals = torch.rand(X.shape[0], device=X.device)

    agree_mask = (sig1 == sig2)
    disagree_mask = (sig1 != sig2)

    # When agreeing, follow the common signal based on p_agree
    y[agree_mask] = torch.where(rand_vals[agree_mask] < p_agree, sig1[agree_mask], 1.0 - sig1[agree_mask])

    # When disagreeing, follow the primary signal based on p_disagree
    y[disagree_mask] = torch.where(rand_vals[disagree_mask] < p_disagree, sig1[disagree_mask],
                                   1.0 - sig1[disagree_mask])

    return y.unsqueeze(1)


RULES_REGISTRY = {
    "upper_half": upper_half_rule,
    "lower_half": lower_half_rule,
    "parity": parity_rule,
    "probabilistic_shortcut": probabilistic_shortcut
}