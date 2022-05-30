import pandas as pd
import os
import itertools as itls
from .dict_tools import init_cols


def change_dframe_labels(dframe, col, old_labels=None, new_labels=None):
    # runs through old labels and replaces with new labels
    label_list = dframe[col].to_list()  # change label names in list external to dframe
    for idx, i_label in enumerate(label_list):
        for jdx in range(len(old_labels)):
            if i_label == old_labels[jdx]:  # if matches old label
                label_list[idx] = new_labels[jdx]  # replace with matching new label

    output_dframe = dframe.copy()  # preserve original if desired
    og_columns = output_dframe.columns.to_list()  # save original column order for later
    temp_columns = [x for x in og_columns]  # creates unlinked copy of list
    temp_columns.remove(col)  # remove column you wish to replace
    output_dframe = output_dframe[temp_columns]
    output_dframe[col] = label_list  # replace column with new labels
    output_dframe = output_dframe[og_columns]  # re-orders columns to original order

    return output_dframe


def change_dframe_colnames(dframe, old_colnames, new_colnames):
    """ change multiple col names """
    og_colnames = dframe.columns.to_list()
    updated_colnames = [col for col in og_colnames]  # create copy of og_cols in og order
    output_dframe = dframe.copy()
    for idx in range(len(old_colnames)):
        old = old_colnames[idx]
        new = new_colnames[idx]
        for col_idx, col in enumerate(og_colnames):
            if col == old:
                updated_colnames[col_idx] = new  # replace label in list of colnames in order
                output_dframe[new] = dframe[old]  # add copy of og_column under new colname
    output_dframe = output_dframe[updated_colnames]  # use only these colnames
    return output_dframe


def index_dframe(dframe, columns, values, andor='and'):
    # indexes dframe where col == val
    if len(columns) == 1 and len(values) > 1:
        columns = itls.repeat(columns[0])  # no need to repeat if single column
    output_dframe = None
    if andor == 'or':
        # each condition can be True indepednent of other conds
        output_dframe = []
        # if from same columns
        for i_col, i_val in zip(columns, values):
            output_dframe.append(dframe[(dframe[i_col] == i_val)])
        output_dframe = pd.concat(output_dframe)
    elif andor == 'and':
        # all conditions are True together
        output_dframe = dframe.copy()
        for i_col, i_val in zip(columns, values):
            output_dframe = output_dframe[(output_dframe[i_col] == i_val)]
    return output_dframe


def to_csv_pkl(dframe, folder, title, rnd=2, _csv=True, _pkl=True):
    if _csv is True:
        dframe.round(rnd).to_csv(os.path.join(folder, f"{title}.csv"), index=False)
    if _pkl is True:
        dframe.to_pickle(os.path.join(folder, f"{title}.pkl"))


def cols2rows(dframe, change_cols, new_col):
    # useful to convert summary_data data tables for plotting
    keep_cols = [i for i in dframe.columns if all(j != i for j in change_cols)]
    output = init_cols(keep_cols + [new_col] + ['values'])
    for i_change in change_cols:
        output[new_col] += [i_change] * len(dframe)
        output['values'] += dframe[i_change].to_list()
        for i_keep in keep_cols:
            output[i_keep] += dframe[i_keep].to_list()

    return pd.DataFrame(output)


def remove_rows(dframe, col, vals):
    rows2remove = index_dframe(dframe, columns=[col], values=vals, andor='or').index
    dframe = dframe.drop(index=rows2remove)
    return dframe


def custom_sort(dframe, cols, ordered_labels):
    # sort selected cols of dframe based on custom order
    _dframe = dframe.copy()
    numbered_orders = {}
    for col, order in zip(cols, ordered_labels):
        for jdx, cond in enumerate(order):
            numbered_orders[cond] = jdx
    _dframe = _dframe.sort_values(by=cols, key=lambda x: x.map(numbered_orders))
    return _dframe
