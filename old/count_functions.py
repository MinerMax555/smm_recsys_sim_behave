import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def prepare_counts_import(inters: pd.DataFrame, tracks: pd.DataFrame, users: pd.DataFrame, LEs=False):
    res = dict()

    u_countries = list(users['country'].unique())
    i_countries = list(tracks['country'].unique())

    for u_c in u_countries:
        res[u_c] = dict()
        for i_c in i_countries:
            res[u_c][i_c] = dict()
            # if LEs: res[u_c][i_c]['LEs'] = 0
            res[u_c][i_c]['unique'] = 0

    # for u_c in tqdm(u_countries):
    for u_c in u_countries:

        u_c_users = np.array(users[users['country'] == u_c].index)
        u_c_inters = inters[inters.uid.isin(u_c_users)]

        total_unique_inters = len(u_c_inters)

        joined = u_c_inters.join(tracks, on='tid')
        # if LEs: LE_distr = joined.groupby('country')['LEs'].sum()
        cnt_distr = joined.groupby('country')['tid'].count()

        for i_c in cnt_distr.index:
            # if LEs: res[u_c][i_c]['LEs'] += LE_distr[i_c]
            res[u_c][i_c]['unique'] += cnt_distr[i_c]

        # res[u_c]['LEs_total'] = total_LEs
        # res[u_c]['unique_total'] = total_unique_inters

    return res


def prepare_counts_track_export(inters: pd.DataFrame, tracks: pd.DataFrame, users: pd.DataFrame, LEs=False):
    res = dict()

    i_countries = list(tracks['country'].unique())
    u_countries = list(users['country'].unique())

    for i_c in i_countries:
        res[i_c] = dict()
        for u_c in u_countries:
            res[i_c][u_c] = dict()
            if LEs: res[i_c][u_c]['LEs'] = 0
            res[i_c][u_c]['unique'] = 0

    for i_c in tqdm(i_countries):

        i_c_tracks = np.array(tracks[tracks['country'] == i_c].index)
        i_c_inters = inters[inters.tid.isin(i_c_tracks)]

        total_unique_inters = len(i_c_inters)
        if LEs: total_LEs = i_c_inters.LEs.sum()

        joined = i_c_inters.join(users, on='uid')
        if LEs: LE_distr = joined.groupby('country')['LEs'].sum()
        cnt_distr = joined.groupby('country')['uid'].count()

        for u_c in cnt_distr.index:
            if LEs: res[i_c][u_c]['LEs'] += LE_distr[u_c]
            res[i_c][u_c]['unique'] += cnt_distr[u_c]

        # res[u_c]['LEs_total'] = total_LEs
        # res[u_c]['unique_total'] = total_unique_inters

    return res


def consumption_distribution(target: str, cat_lands: [str], counts: dict, LEs=False):
    cats = cat_lands
    if not target in counts[target].keys():
        print('[Warning]: local music not appreciated or key error')

    if 'domestic' in cats:
        cats.remove('domestic')
        if not target in cats:
            cats += [target]

    if not 'other' in cats:
        cats += ['other']

    if not 'total' in cats:
        cats += ['total']

    res = dict()
    for c in cats:
        res[c] = dict()
        res[c]['unique'] = 0
        if LEs: res[c]['LEs'] = 0

    for k in counts[target].keys():
        u_count = counts[target][k]['unique']
        if LEs: le_count = counts[target][k]['LEs']

        res['total']['unique'] += u_count
        if LEs: res['total']['LEs'] += le_count

        if k in cats:
            res[k]['unique'] += u_count
            if LEs: res[k]['LEs'] += le_count
        else:
            res['other']['unique'] += u_count
            if LEs: res['other']['LEs'] += le_count

    return res


## export - domestic, US consumption, other
def viz_local_us_other_import_export(targets_0: [str], counts: dict, save_as='', mode='LEs', proportions=False,
                                     average=False, imported=True):
    average_US = 0
    average_local = 0

    overall_local = 0
    targets = np.array(targets_0)

    labels = None
    if imported:
        labels = ['US music', 'other', 'domestic music']
    else:
        labels = ['US consumption', 'other', 'domestic']

    # colors = ['#992201','#FF5511', '#44BBDD']
    colors = ['#ff7a62', '#d3d3d3', '#0571b0']
    # colors = ['#ff7760','#e5c1c1', '#0571b0']

    data = np.zeros((len(targets), 3))

    for i, t in zip(range(len(targets)), targets):
        distr = consumption_distribution(target=t,
                                         cat_lands=['US', 'other', 'domestic'],
                                         counts=counts,
                                         LEs=True if mode == 'LEs' else False)

        data[i, 0] = 0 if t == 'US' else distr['US'][mode]
        data[i, 1] = distr['other'][mode]
        data[i, 2] = distr[t][mode]

        if proportions:
            data[i, :] = data[i, :] / data[i, :].sum()

    # sorting rows
    dom = data[:, 0]
    new_order = dom.argsort()
    data = data[new_order, :]
    targets = list(targets[new_order])

    # calculating average US and local values
    if proportions:
        average_US = data[:-1, 0].mean()
        average_local = data[:-1, 2].mean()
    else:
        average_US = data[:, 0].mean()
        average_local = data[:, 2].mean()
    # average_local = overall_local / len(targets)

    out_targets = targets.copy()
    if average and proportions:
        a_row = [average_US, 0, average_local]
        data = np.vstack([data, a_row])
        out_targets += ['avg']

    width = 0.4

    fig, ax = plt.subplots()

    # Viz params
    # fig.set_size_inches(8.5, 6.5)
    fig.set_size_inches(8., 9.)

    plt.margins(0, 0, tight=True)

    SMALL_SIZE = 18
    MEDIUM_SIZE = 18

    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    # rendering
    left = np.zeros(data.shape[0])

    for i, l, c in zip(range(3), labels, colors):
        if average and proportions and (l == 'domestic music' or l == 'domestic consumption'):
            left[-1] = 1 - data[-1, 2]
        ax.barh(out_targets, data[:, i], color=c, left=left, label=l)
        left += data[:, i]

    # baselines
    if average and proportions:
        plt.axvline(x=average_US, color='#992201')
        plt.axvline(x=1 - average_local, color='#0241b0')

    xlabel = ""
    if mode == "LEs":
        xlabel = "Listening Events"
    elif mode == "unique":
        xlabel = "Interactions"

    if proportions:
        xlabel = "Proportion of " + xlabel

    ylabel = ""
    if imported:
        ylabel = "User Country"
    else:
        ylabel = "Artist Country"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('')
    # ax.legend()

    ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.02), loc=3,
              ncol=3, mode="expand", borderaxespad=0., frameon=False, handletextpad=0.3)
    if '.' in save_as:
        plt.savefig(save_as)

    plt.show()

    return data


def viz_import_export_REC(targets_0: [str], counts1: dict, counts2: dict, save_as='', mode='LEs', proportions=False,
                          average=False, imported=True):
    average_US1 = 0
    average_local1 = 0

    average_US2 = 0
    average_local2 = 0

    targets = np.array(targets_0)

    labels = None
    if imported:
        labels1 = ['consmd. US music', '', 'consmd. domestic']
        labels2 = ['rec. US music', '', 'rec. domestic']
    else:
        labels1 = ['US consumption', '', 'local consumptions']
        labels2 = ['rec. to US users', '', 'rec. to local']

    # colors = ['#992201','#FF5511', '#44BBDD']
    colors1 = ['#992201', '#FFFFFF', '#0571b0']
    colors2 = ['#ff7760', '#FFFFFF', '#0571b0']

    data1 = np.zeros((len(targets), 3))
    data2 = np.zeros((len(targets), 3))

    # DATA 1
    for i, t in zip(range(len(targets)), targets):
        distr = consumption_distribution(target=t,
                                         cat_lands=['US', 'other', 'domestic'],
                                         counts=counts1,
                                         LEs=True if mode == 'LEs' else False)

        data1[i, 0] = 0 if t == 'US' else distr['US'][mode]
        data1[i, 1] = distr['other'][mode]
        data1[i, 2] = distr[t][mode]

        if proportions:
            data1[i, :] = data1[i, :] / data1[i, :].sum()

    # DATA 2
    for i, t in zip(range(len(targets)), targets):
        distr = consumption_distribution(target=t,
                                         cat_lands=['US', 'other', 'domestic'],
                                         counts=counts2,
                                         LEs=True if mode == 'LEs' else False)

        data2[i, 0] = 0 if t == 'US' else distr['US'][mode]
        data2[i, 1] = distr['other'][mode]
        data2[i, 2] = distr[t][mode]

        if proportions:
            data2[i, :] = data2[i, :] / data2[i, :].sum()

    dom = data1[:, 0]
    new_order = dom.argsort()
    data1 = data1[new_order, :]
    data2 = data2[new_order, :]
    targets = list(targets[new_order])

    # calculating average US and local values
    if proportions:
        average_US1 = data1[:-1, 0].mean()
        average_local1 = data1[:-1, 2].mean()

        average_US2 = data2[:-1, 0].mean()
        average_local2 = data2[:-1, 2].mean()
    else:
        average_US1 = data1[:, 0].mean()
        average_local1 = data1[:, 2].mean()

        average_US2 = data2[:, 0].mean()
        average_local2 = data2[:, 2].mean()
    # average_local = overall_local / len(targets)

    out_targets = targets.copy()
    if average and proportions:
        a_row1 = [average_US1, 0, average_local1]
        a_row2 = [average_US2, 0, average_local2]

        data1 = np.vstack([data1, a_row1])
        data2 = np.vstack([data2, a_row2])
        out_targets += ['avg']

    width = 0.4

    fig, ax = plt.subplots()

    # Viz params
    # fig.set_size_inches(8.5, 6.5)
    fig.set_size_inches(8., 9.)

    plt.margins(0.00, 0.01, tight=True)

    SMALL_SIZE = 18
    MEDIUM_SIZE = 18

    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    # rendering
    left1 = np.zeros(data1.shape[0])
    left2 = np.zeros(data2.shape[0])

    # baselines
    if average and proportions:
        # plt.axvline(x = average_US1, color = '#992201', alpha = 0.7, linewidth = 2)
        # plt.axvline(x = 1 - average_local1, color = '#0571b0', alpha = 0.9, linewidth = 2)

        plt.axvline(x=average_US2, color='#992201', alpha=0.4, linewidth=2)
        plt.axvline(x=1 - average_local2, color='#0571b0', alpha=0.4, linewidth=2)

    for i, l1, l2, c1, c2 in zip(range(3), labels1, labels2, colors1, colors2):
        if average and proportions and ('domestic' in l1 or 'local' in l1):
            left1[-1] = 1 - data1[-1, 2]
            left2[-1] = 1 - data2[-1, 2]

        ax.barh(out_targets, data2[:, i], color=c2, left=left2, label=l2,
                alpha=0.7, linewidth=0, fill=(True if l2 != '' else False), height=0.65)
        ax.barh(out_targets, data1[:, i], edgecolor=c1, left=left1, label=l1,
                alpha=1.0, linewidth=(2 if l1 != '' else 0), fill=False, height=0.7)

        left1 += data1[:, i]
        left2 += data2[:, i]

    xlabel = ""
    if mode == "LEs":
        xlabel = "Listening Events"
    elif mode == "unique":
        xlabel = "Interactions"

    if proportions:
        xlabel = "Proportion of " + xlabel

    ylabel = ""
    if imported:
        ylabel = "User Country"
    else:
        ylabel = "Artist Country"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('')
    # ax.legend()

    ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.02), loc=3,
              ncol=2, mode="expand", borderaxespad=0., frameon=False, handletextpad=0.3)
    if '.' in save_as:
        plt.savefig(save_as)

    plt.show()

    return  # data