from copy import copy


def init_cols(cols, val=None):
    if val is None:
        val = []
    output = {}
    for i_col in cols:
        output[i_col] = copy(val)  # copy used to keep vals independent
    return output


def append_dicty(dicty, values):
    if len(dicty.keys()) != len(values):
        raise ValueError("Length of dict and values must be equal")
    for idx, key in enumerate(dicty):
        dicty[key].append(values[idx])
        