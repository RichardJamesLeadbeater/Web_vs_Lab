import pandas as pd
import datetime


def add_cardinal_and_oblique(dframe, ori_label):
    ori_ = ori_label
    cardinals = dframe[(dframe[ori_] == 'horizontal') | (dframe[ori_] == 'vertical')].copy()
    cardinals[ori_] = 'cardinal'
    obliques = dframe[(dframe[ori_] == 'minus45') | (dframe[ori_] == 'plus45')].copy()
    obliques[ori_] = 'oblique'
    output_dframe = pd.concat([dframe, cardinals, obliques])
    return output_dframe


def get_tstamp():
    t_now = datetime.datetime.now()  # create timestamp
    t_stamp = [t_now.year, t_now.month, t_now.day, t_now.hour, t_now.minute]
    for idx, t in enumerate(t_stamp):
        if len(str(t)) == 1:
            t_stamp[idx] = f"0{t}"  # adds 0 to single digit nums (aids sorting)
    return ''.join([str(t) for t in t_stamp])
