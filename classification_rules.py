def upper_half_rule(names, features_types, deciding_feature=0, **kwargs):
    threshold = features_types[deciding_feature] // 2
    return (names[:, deciding_feature] < threshold).float()

def parity_rule(names, deciding_feature=0, **kwargs):
    return (names[:, deciding_feature] % 2 == 0).float()

RULES_REGISTRY = {
    "upper_half": upper_half_rule,
    "parity": parity_rule
}

