import pickle as pkl
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn import metrics
import networkx as nx

sys.path.append("../nextrout_tutorial/")
import utils

sys.path.append("./networks/synthetic_data_SBM/")
import SBM_generation as sbm
import os

from matplotlib.lines import Line2D

"""
A python module to generate nice plots
"""
new_markers = [
    ",",
    "o",
    "v",
    # ">",
    "^",
    # "8",
    # "s",
    "p",
    "*",
    "H",
    "D",
    # 4,
    # "_",
    "X",
    # "P",
    # "X",
    # 9,
    # 10,
    # 11,
    (5, 0, 20),
    (7, 0, 20),
    # (5, 1, 20),
    (7, 0, 30),
    (6, 1, 45),
    (4, 1, 30),
    (8, 1, 20),
    (3, 1, 45),
    (2, 1, 45),
    (2, 1, 0),
    (3, 1, 0),
    (10, 1, 0),
    (20, 1, 0),
    (15, 1, 0),
    # "x",
]

greens = [
    "#543005",
    "#8c510a",
    "#bf812d",
    "#dfc27d",
    "#80cdc1",
    "#35978f",
    "#01665e",
    # "#003c30",
]
pinks = [
    "#8e0152",
    "#c51b7d",
    "#de77ae",
    "#f1b6da",
    "#fde0ef",
    "#e6f5d0",
    "#b8e186",
    "#7fbc41",
    "#4d9221",
    # "#276419",
]
purples = [
    # "#40004b",
    "#762a83",
    "#9970ab",
    "#c2a5cf",
    # "#e7d4e8",
    "#d9f0d3",
    "#a6dba0",
    "#5aae61",
    "#1b7837",
    # "#00441b",
]
browns = [
    "#7f3b08",
    "#b35806",
    "#e08214",
    "#fdb863",
    "#fee0b6",
    "#d8daeb",
    "#b2abd2",
    "#8073ac",
    "#542788",
    # "#2d004b",
]
reds = [
    "#67001f",
    "#b2182b",
    "#d6604d",
    "#f4a582",
    "#fddbc7",
    "#d1e5f0",
    "#92c5de",
    "#4393c3",
    "#2166ac",
    # "#053061",
]
blues = [
    "#a50026",
    "#d73027",
    "#f46d43",
    "#fdae61",
    "#fee090",
    "#e0f3f8",
    "#abd9e9",
    "#74add1",
    "#4575b4",
    # "#313695",
]

cmap = greens + pinks + purples + browns + reds + blues
# cmap = (greens[2:-2]+pinks[1:-2]+purples[1:-2]+browns[2:-2]+reds[1:-2]+blues[2:-2])
random.seed(8)  # 4 for american football, 1 for the other methods
colours = random.sample(cmap, 50)

b = [
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
]
r = [
    "#fdd49e",
    "#fdbb84",
    "#fc8d59",
    "#e34a33",
    "#b30000",
]

other_methods = [
    "#41ae76",  # green
    "#de2d26",  # red
    "#fdb863",  # yellow-ish
    # '#bf812d', #brown-ish
    # '#df65b0', # yellow-ish
    "#8c6bb1",  # purple
]
cores = other_methods + b


def set_markers_gt(G, communities, gt=False):
    # markers = list(Line2D.markers)[1:]

    markers = new_markers
    ncomm = list(set(dict(communities).values()))
    communities = dict(communities)
    markers_list = markers[: len(ncomm)]

    mapping = dict()
    for i, j in list(zip(ncomm, markers_list)):
        mapping[i] = j

    colours_list = colours[: len(ncomm)]
    color_mapping = dict()
    for i, j in list(zip(ncomm, colours_list)):
        color_mapping[i] = j

    node_markers = {}
    for node, value in communities.items():
        node_markers[node] = mapping[value]

    node_colours = {}
    for node, value in communities.items():
        node_colours[node] = color_mapping[value]

    if gt:
        for node in G.nodes():
            nx.set_node_attributes(G, node_markers, "marker")

    if gt:
        for node in G.nodes():
            nx.set_node_attributes(G, node_colours, "colour")

    return G


def communities2dict(communities, community_mapping):
    """Finds a mapping for similar communities between orig and inferred ones"""

    original = set(dict(communities).values())
    inferred = set(dict(community_mapping).values())

    # we use a list comprehension, iterating through keys and checking the values that match each n

    comm_orig = {}
    for n in original:
        comm_orig[n] = [
            k for k in dict(communities).keys() if dict(communities)[k] == n
        ]

    comm_met = {}
    for n in inferred:
        comm_met[n] = [
            k for k in dict(community_mapping).keys() if dict(community_mapping)[k] == n
        ]

    return comm_orig, comm_met


def node_colouring(G, G2, communities, community_mapping, ndiff, synt=False):
    ncomm = list(set(dict(communities).values()))
    if synt:
        colours_list = colours[:]
    else:
        colours_list = colours[len(ncomm) + 1 :]
    #####
    comm_orig, comm_met = communities2dict(communities, community_mapping)
    # G = set_markers_gt(G, communities, gt=True)
    gt = G.copy()
    equal_comms = {}
    diff_comms = {}

    """Find the nodes corresponding to the same communities as the gt"""
    for i, j in comm_orig.items():
        for m, n in comm_met.items():
            if len(j) <= (len(n) - ndiff):
                equal_comms[m] = n
            elif sorted(j) == sorted(n):
                equal_comms[m] = n

    same_colour = list()
    for i, j in equal_comms.items():
        for m in j:
            same_colour.append(m)

    """Find isolated nodes and smaller communities not corresponding to the gt"""

    diff_comms = dict()
    isolated_nodes = []
    for k, v in comm_met.items():
        if k not in equal_comms.keys():
            if len(v) == 1:
                isolated_nodes.append(v)
            else:
                diff_comms[k] = v

    """Colouring smaller communities"""
    smaller_comms = {}
    index = -1
    for k in diff_comms.keys():
        index += 1
        if colours[index] not in set(nx.get_node_attributes(G2, "colour").values()):
            smaller_comms[k] = colours_list[index]

    final = {}
    for i in diff_comms.keys():
        for node in diff_comms[i]:
            final[node] = smaller_comms[i]

    node_colours_inferred = {}
    for node in G2.nodes():
        if node in same_colour:
            node_colours_inferred[node] = nx.get_node_attributes(gt, "colour")[node]
            G2.nodes[node]["marker"] = "o"

        else:
            G2.nodes[node]["marker"] = "s"
            if node in final.keys():
                node_colours_inferred[node] = final[node]
            else:
                node_colours_inferred[node] = "black"

    return node_colours_inferred, isolated_nodes, G2


def node_colouring_others(G, G2, communities, community_mapping, ndiff):
    ncomm = list(set(dict(communities).values()))
    colours_list = colours[:]
    #####
    comm_orig, comm_met = communities2dict(communities, community_mapping)
    # G = set_markers_gt(G, communities, gt=True)
    gt = G.copy()
    equal_comms = {}
    diff_comms = {}
    sc = list()
    same_colour = list()

    """Finds the nodes corresponding to the same communities as the gt"""
    for i, j in comm_orig.items():
        for m, n in comm_met.items():
            if abs(len(j) - len(n)) <= ndiff:
                equal_comms[m] = n
                sc.append(list(set(j).intersection(n)))

    for i in sc:
        if len(i) > 1:
            for j in i:
                same_colour.append(j)

    """Find isolated nodes and smaller communities not corresponding to the gt"""

    diff_comms = dict()
    isolated_nodes = []
    for k, v in comm_met.items():
        if k not in equal_comms.keys():
            if len(v) == 1:
                isolated_nodes.append(v)
            else:
                diff_comms[k] = v

    """Colouring smaller communities"""
    smaller_comms = {}
    index = -1
    for k in comm_met.keys():
        index += 1
        # if colours[index] not in set(nx.get_node_attributes(G2, "colour").values()):
        smaller_comms[k] = colours[index]

    final = {}
    for i, j in comm_met.items():
        for node in j:
            final[node] = smaller_comms[i]

    intersections = {}
    for i, j in comm_orig.items():
        len_intersection = []
        for m, n in comm_met.items():
            len_intersection.append((m, len(set(j).intersection(n))))
            # print(i,m,set(j).intersection(n))

        len_intersection.sort(key=lambda x: x[1])
        intersections[i] = len_intersection[-1]
    same_colour = []
    for i in intersections.keys():
        same_colour += comm_met[intersections[i][0]]

    different_com = {}
    for idx, k in enumerate(comm_met.keys()):
        for i in intersections.keys():
            if k != intersections[i][0]:
                # print(k,len(colours),idx+len(ncomm)+1)
                different_com[k] = colours[idx + len(ncomm) + 1]

    node_colours_inferred = {}
    for node in G2.nodes():
        if node in same_colour:
            node_colours_inferred[node] = nx.get_node_attributes(gt, "colour")[node]
            G2.nodes[node]["marker"] = "o"
        else:

            node_colours_inferred[node] = different_com[community_mapping[node]]
            G2.nodes[node]["marker"] = "s"

    return node_colours_inferred, G2


def node_markers(G, G_, communities, community_mapping):

    comm_orig, comm_met = communities2dict(communities, community_mapping)
    gt = set_markers_gt(G, communities, gt=True)

    equal_comms = {}
    diff_comms = {}

    """Find the nodes corresponding to the same communities as the gt"""
    for i, j in comm_orig.items():
        for m, n in comm_met.items():
            if len(j) <= (len(n) - 3):
                equal_comms[m] = n
            elif j == n:
                equal_comms[m] = n

    same_markers = list()
    for i, j in equal_comms.items():
        for m in j:
            same_markers.append(m)

    """Find isolated nodes and smaller communities not corresponding to the gt"""

    diff_comms = dict()
    isolated_nodes = []
    for k, v in comm_met.items():
        if k not in equal_comms.keys():
            if len(v) == 1:
                isolated_nodes.append(v)
            else:
                diff_comms[k] = v

    markers = new_markers

    smaller_comms = {}
    index = -1
    for k in diff_comms.keys():
        index += 1
        if markers[index] not in set(nx.get_node_attributes(G_, "marker").values()):
            smaller_comms[k] = markers[index]

    final = {}
    for i in diff_comms.keys():
        for node in diff_comms[i]:
            final[node] = smaller_comms[i]

    node_markers_inferred = {}
    for node in G_.nodes():
        if node in same_markers:
            node_markers_inferred[node] = nx.get_node_attributes(gt, "marker")[node]
        else:
            if node in final.keys():
                node_markers_inferred[node] = final[node]
            else:
                node_markers_inferred[node] = "X"

    return node_markers_inferred


def plot_with_markers(
    original_G,
    communities,
    outputs,
    method_list,
    metric,
    comm_label,
    wt,
    alpha,
    ns,
    dataset,
    labels=False,
    seed=0,
):

    pos = nx.spring_layout(original_G, seed=seed)
    fig, ax = plt.subplots(
        figsize=(10 * len(method_list), 8), nrows=1, ncols=len(method_list)
    )

    i = -1

    G = set_markers_gt(original_G, communities, gt=True)
    nodeShapes = set((aShape[1]["marker"] for aShape in G.nodes(data=True)))

    for col in ax:
        i += 1
        method = method_list[i]
        print(method)
        labels_true = []
        predicted_labels = []
        if method == "ground_truth":
            nodeShapes = set((aShape[1]["marker"] for aShape in G.nodes(data=True)))
            nodecolours = set((acolour[1]["colour"] for acolour in G.nodes(data=True)))
            for shape in {"o"}:  # set(nodeShapes):
                node_list = [
                    node for node in G.nodes()
                ]  # if G.nodes[node]["marker"] == shape]

                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=node_list,
                    node_color=[G.nodes[node]["colour"] for node in node_list],
                    node_shape=shape,
                    edgecolors="black",
                    ax=ax[0],
                    node_size=ns,
                )
            ax[0].axis("off")
            nb_comm = len(set(dict(communities).values()))

            col.set_title(
                "GT" + "\n" + str(nb_comm) + " communities", fontsize=20,
            )
            if labels:
                nx.draw_networkx_labels(G, pos, ax=ax[0])
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax[0],
                alpha=alpha,
                edge_color="gray",
                connectionstyle=("arc3", "rad=0.3"),
            )

        else:

            OT_met = [
                "Nextrout_beta=1.5_induced_False",
                "Nextrout_beta=1.0_induced_False",
                "Nextrout_beta=2.0_induced_False" "OTDSinkhornMix",
            ]
            G_, community_mapping, _ = outputs[method]
            new_G = G_.copy()
            nx.set_node_attributes(new_G, pos, "pos")
            if method not in OT_met:
                ndiff = 0
            else:
                ndiff = 3

            # node_colours_inferred, _ = node_colouring(
            #     G, G_, communities, community_mapping, ndiff
            # )
            # nx.set_node_attributes(new_G, node_colours_inferred, "colour")

            # markers_inferred = node_markers(G, G_, communities, community_mapping)
            # nx.set_node_attributes(new_G, markers_inferred, "marker")

            new_G = set_markers_gt(new_G, community_mapping, gt=True)
            nodeShapes = set((aShape[1]["marker"] for aShape in new_G.nodes(data=True)))
            print(nodeShapes)
            nodecolours = set(
                (acolour[1]["colour"] for acolour in new_G.nodes(data=True))
            )

            community_mapping_new = utils.comm_permutation(
                dict(communities), community_mapping
            )

            community_mapping_new = (
                community_mapping  # comment this line for real networks
            )

            missed_list = []
            correct_list = []

            for node in community_mapping_new.keys():
                if community_mapping_new[node] != communities[node]:
                    missed_list.append(node)
                else:
                    correct_list.append(node)
            for node in community_mapping_new.keys():

                if node in correct_list:
                    marker = "o"
                else:
                    marker = "o"
                # checking whether it is isolated
                if (
                    len(
                        [
                            n
                            for n in community_mapping_new.keys()
                            if community_mapping_new[node] == community_mapping_new[n]
                        ]
                    )
                    == 1
                ):
                    color = "black"
                else:
                    color = G.nodes[node]["colour"]

                nx.draw_networkx_nodes(
                    new_G,
                    pos,
                    nodelist=[node],  # node_list,
                    # node_color=[color],#for node in node_list],
                    node_color=[new_G.nodes[node]["colour"] for node in node_list],
                    node_shape=marker,
                    edgecolors="black",
                    ax=col,
                    node_size=ns,
                )

            nbr_comm = len(set(community_mapping.values()))

            if "Nextrout" in method:
                method = "ORC-Nextrout"
            elif "label" in method:
                method = "Label Propagation"
            elif "Sink" in method:
                method = "OTDSinkhorn"
            else:
                method = method

            col.set_title(
                method.split("_")[0] + "\n" + str(nbr_comm) + " communities",
                # + "; "
                # + metric
                # + "="
                # + str(value),
                fontsize=20,
            )

            # value = utils.metric_comp(
            #     original_G, community_mapping, metric, clustering_label=comm_label
            # )
            # value = round(value, 3)

            col.axis("off")
            nx.draw_networkx_edges(
                new_G,
                pos,
                ax=col,
                alpha=alpha,
                edge_color="gray",
                connectionstyle=("arc3", "rad=0.3"),
            )

            if labels:
                nx.draw_networkx_labels(new_G, pos, ax=col)

    # fig.savefig(
    #     "./plots/markers_" +str(dataset) +'_'
    #     + str(len(method_list))
    #     + "_metric:"
    #     + metric
    #     + ".pdf"
    # )

    fig.tight_layout()


def plot_no_markers(
    original_G,
    communities,
    outputs,
    method_list,
    metric,
    comm_label,
    wt,
    alpha,
    ns,
    dataset,
    seed,
    synt=False,
):
    original_G = set_markers_gt(original_G, communities, gt=True)
    pos = nx.spring_layout(original_G, seed=seed)
    nx.set_node_attributes(original_G, pos, "pos")
    size = 8
    fig, ax = plt.subplots(
        figsize=(size * len(method_list), size - 2), nrows=1, ncols=len(method_list)
    )

    i = -1
    sns.set_context("paper")
    for col in ax:
        i += 1
        method = method_list[i]
        print(method)
        labels_true = []
        predicted_labels = []
        # G = set_markers_gt(original_G, communities, gt=True)
        if method == "ground_truth":
            # G = set_markers_gt(original_G, communities, gt=True)
            # nodecolours = set((acolour[1]["colour"] for acolour in G.nodes(data = True)))
            layout = dict((n, original_G.nodes[n]["pos"]) for n in original_G.nodes())

            nodes = nx.draw_networkx_nodes(
                original_G,
                pos=layout,
                node_color=[original_G.nodes[node]["colour"] for node in original_G.nodes()],
                edgecolors="black",
                ax=ax[0],
                node_size=ns,
                # alpha=0.2,
            )

            for edge in original_G.edges():
                ax[0].annotate(
                    "",
                    xy=layout[edge[0]],
                    xycoords="data",
                    xytext=layout[edge[1]],
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-",
                        color="0.7",
                        shrinkA=8,
                        shrinkB=8,
                        patchA=None,
                        patchB=None,
                        connectionstyle="arc3,rad=0.15",
                    ),
                )

            nodes.set_zorder(10)
            nb_comm = len(set(dict(communities).values()))

            col.set_title(
                "GT" + "\n" + str(nb_comm) + " communities", fontsize=20,
            )
            ax[0].axis("off")
            # nx.draw_networkx_labels(G, pos, ax=ax[0])

        else:
            # OT_met = [
            #     "Nextrout_beta=0.1_induced_False",
            #     "Nextrout_beta=0.5_induced_False",
            #     "Nextrout_beta=1.5_induced_False",
            #     "Nextrout_beta=1.0_induced_False",
            #     "Nextrout_beta=2.0_induced_False",
            #     "OTDSinkhornMix",
            #     "Nextrout"
            # ]
            new_G, community_mapping, _ = outputs[method]

            nx.set_node_attributes(new_G, pos, "pos")
            if not ( 'Nextrout' in method or 'OTD' in method or 'Sinkhorn' in method):
                if synt:
                    node_colours_inferred, new_G = set_squares_synt(
                        original_G, new_G, communities, community_mapping
                    )

                else:
                    if "les" in dataset:
                        node_colours_inferred, new_G = markers_for_real_nets_infomap(
                            original_G, new_G, communities, community_mapping
                        )

                    else:

                        node_colours_inferred, new_G = node_colouring_others(
                            original_G, new_G, communities, community_mapping, ndiff=3
                        )
            else:
                if synt:
                    node_colours_inferred, new_G = set_squares_synt(
                        original_G, new_G, communities, community_mapping
                    )
                else:
                    node_colours_inferred, new_G = markers_for_real_nets(
                        original_G, new_G, communities, community_mapping
                    )

                    # node_colours_inferred, new_G = set_squares_synt(
                    #     G, new_G, communities, community_mapping
                    # )

            nx.set_node_attributes(new_G, node_colours_inferred, "colour")

            layout = dict((n, new_G.nodes[n]["pos"]) for n in original_G.nodes())
            # community_mapping_new = utils.comm_permutation(dict(communities), community_mapping)
            community_mapping_new = community_mapping
            missed_list = []
            correct_list = []

            for node in community_mapping_new.keys():
                # print(community_mapping_new[node],communities[node])
                if community_mapping_new[node] != communities[node]:
                    missed_list.append(node)
                else:
                    correct_list.append(node)

            # square = [10, 21, 27, 28, 29, 43, 99, 100, 107, 130, 143, 152, 162, 163, 165, 168, 173, 179, 180, 183, 209, 212, 215, 221, 265, 271, 274, 277, 279, 289, 291, 294, 305, 308, 317, 325, 328, 330, 332, 334, 348, 355, 359, 360, 367, 370, 381, 399, 405, 409, 416, 425, 432, 434, 447, 448, 451, 460, 461, 466, 470, 472, 474, 476, 480, 483, 484, 486, 488, 489]

            for node in community_mapping_new.keys():
                # if node in square and method=='OTDSinkhornMix' and 'LFR' in dataset :
                #     marker='s'
                # else:
                #     marker='o'
                # if node in correct_list:marker = 'o'
                # else: marker = 's'
                # checking whether it is isolated
                if (
                    len(
                        [
                            n
                            for n in community_mapping_new.keys()
                            if community_mapping_new[node] == community_mapping_new[n]
                        ]
                    )
                    == 1
                ):
                    color = "black"
                    marker = "s"
                else:
                    color = new_G.nodes[node]["colour"]
                # try:
                marker = new_G.nodes[node]["marker"]
                import numpy as np

                # print(
                #     np.sum(
                #         np.array(list(dict(new_G.nodes(data="marker")).values())) == "s"
                #     )
                # )
                # except:  # if ground truth net
                # marker = "o"
                nodes = nx.draw_networkx_nodes(
                    new_G,
                    layout,  # pos,
                    nodelist=[node],  # node_list,
                    node_color=[color],  # for node in node_list],
                    # node_color=[new_G.nodes[node]["colour"] for node in node_list],
                    node_shape=marker,
                    edgecolors="black",
                    ax=col,
                    node_size=ns,
                    # alpha=0.2,
                )
                nodes.set_zorder(10)

            # nodes = nx.draw_networkx_nodes(
            #     new_G,
            #     pos=layout,
            #     node_color=[new_G.nodes[node]["colour"] for node in new_G.nodes()],
            #     edgecolors="black",
            #     ax=col,
            #     node_size=ns,
            # )
            # nx.draw_networkx_edges(
            #     new_G,
            #      pos=layout,
            # #     node_color=[new_G.nodes[node]["colour"] for node in new_G.nodes()],
            #      edge_color="gray",
            #      ax=col,
            # #     node_size=ns,
            #  )

            for edge in new_G.edges():
                col.annotate(
                    "",
                    xy=layout[edge[0]],
                    xycoords="data",
                    xytext=layout[edge[1]],
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-",
                        color="0.7",
                        shrinkA=8,
                        shrinkB=8,
                        patchA=None,
                        patchB=None,
                        connectionstyle="arc3,rad=0.15",
                    ),
                )

            nodes.set_zorder(10)
            # nx.draw_networkx_labels(new_G, pos, ax=col)
            nbr_comm = len(set(community_mapping.values()))
            # value = utils.metric_comp(
            #     original_G, community_mapping, metric, clustering_label=comm_label
            # )
            # value = round(value, 3)
            if "Nextrout" in method:
                method = "ORC-Nextrout"
            elif "label" in method:
                method = "Label Propagation"
            elif "Sink" in method:
                method = "OTDSinkhorn"
            else:
                method = method
            col.set_title(
                method.split("_")[0] + "\n" + str(nbr_comm) + " communities",
                # + "; "
                # + metric
                # + "="
                # + str(value),
                fontsize=20,
            )
            col.axis("off")
    fig.savefig(
        "../data/plots/colours_sside_nnets:"
        + str(dataset)
        + str(len(method_list))
        + "_metric:"
        + metric
        + ".jpeg"
    )

    fig.tight_layout()


def line_plots_sbm(
    df,
    graph_model_full,
    metric,  # df feature to show in the y axis; normally computed quality metric
    x="param",  # df feature to show in the x axis
    extra=None,  # extra information to be added to the name of the output png file
):
    method_list = sorted(list(set(df["method"])))
    markers_list = ["o", "s", "P", "X", "*", "D", "v", "^", "p"]
    """
    Script to get line plots from df.
    """
    labels = [
        "MT",
        "OTDSinkhorn",
        "Infomap",
        "Label P ropagation",
        "ORC-Nextrout, " + r"$\beta=0.1$",
        "ORC-Nextrout, " + r"$\beta=0.5$",
        "ORC-Nextrout, " + r"$\beta=1.0$",
        "ORC-Nextrout, " + r"$\beta=1.5$",
        "ORC-Nextrout, " + r"$\beta=2.0$",
    ]

    sns.set(rc={"figure.figsize": (7, 6)})
    sns.set_style("ticks")
    sns.set_style({"font.family": "serif", "font.serif": "Arial"})
    sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 3.5})

    sns_plot = sns.lineplot(
        data=df,
        x=x,
        y=metric,
        hue="method",
        palette=cores[0 : len(method_list)],
        style="method",
        linewidth=1.9,
        markers=markers_list,
        markersize=6.0,
        dashes=True,
        err_style="band",
    )

    sns_plot.set(xlabel="r")

    leg = plt.legend(ncol=1, frameon=True, numpoints=1, fontsize=8)
    leg.legendHandles[0]._legmarker.set_markersize(5)
    leg.legendHandles[1]._legmarker.set_markersize(5)

    leg = sns_plot.get_legend()

    new_title = None
    leg.set_title(new_title)

    new_labels = labels
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)

    for line in leg.get_lines():
        line.set_linewidth(2.0)

    fig = sns_plot.get_figure()
    if extra is None:
        extra = "_"

    fig.savefig(
        "../data/plots/new_markers_all_met_err_"
        + graph_model_full
        + "_"
        + metric
        + "_"
        + str(x)
        + extra
        + ".png",
        bbox_inches="tight",
        dpi=300,
        format="png",
    )

    plt.show()


from collections import Counter


def df_for_line_plots(
    graph_model,  # either SBM or LFR
    comm_flag,  # community label; 'block' for SBM, 'community' for LFR
    param_list,  # list of param. used to generate nets
    rseed_list,  # list of random seeds
    method_list,
    metric,  # either ARI or F1
    avg_degree_list,  # list of average degrees used to generate the SBM nets
    K_list,  # list of number of clusters K used to generate the SBM nets
    N_list,  # list of number of network sizes used to generate the SBM nets
    verbose=True,
    max_iter=15,
):

    # @df_for_line_plots
    """
    Script to generate the dataframe for a given set of networks, parameters, methods, including
    quality metric for the obtained community mappings.    
    """
    # initialize container for data
    data = []
    # creating lists to store other useful information
    best_iter_list = []
    rseed_full_list = []
    avg_degree_full_list = []
    K_full_list = []
    N_full_list = []
    eff_avg_degree_list = []
    avg_com_size_list = []

    if "LFR" in graph_model:
        K_list = [2]  # stored like this
        avg_degree_list = [0]  # dummy
        N_list = [500]  # pre assigned

    for met in method_list:
        print(met)
        param_list.sort()

        for par in tqdm(param_list):

            for rseed in rseed_list:

                for avg_degree in avg_degree_list:

                    for K in K_list:

                        for N in N_list:
                            # print('N',N)

                            if "SBM" in graph_model:
                                # defining long graph model name
                                graph_model_full = (
                                    graph_model + "_N" + str(N) + "_K=" + str(K)
                                )
                                # loading G
                                G = utils.get_SBM(
                                    int(N), int(K), int(avg_degree), par, rseed,
                                )
                                # this to be sure that G has continuous integers as labels
                                G = nx.convert_node_labels_to_integers(G)

                            elif "LFR" in graph_model:
                                graph_model_full = graph_model + "_N" + str(N)
                                results_ = open(
                                    "networks/lfr_n_500_original/lfr_nets_dict.pkl",
                                    "rb",
                                )
                                lfr_graphs = pkl.load(results_)
                                G = lfr_graphs[str(par)][str(rseed)]
                            else:
                                print("wrong graph model")

                            d = [G.degree[i] for i in G.nodes()]
                            eff_avg_degree = round(np.mean(d))

                            if "Mix" in met or "Next" in met or "McOpt" in met:

                                if "Next" in met:
                                    if "LFR" not in graph_model_full:
                                        file = (
                                            "./communities/"
                                            + graph_model
                                            + "/K="
                                            + str(K)
                                            + "/"
                                            + met.split("_beta")[0]
                                            + "/beta="
                                            + met.split("_beta=")[1].split("_")[0]
                                            + "/communities_dict_"
                                            + met
                                            + "_num_iter_N_"
                                            + graph_model_full
                                            + "_avg_degree="
                                            + str(avg_degree)
                                            + "_rseed="
                                            + str(rseed)
                                            + "_a="
                                            + str(par)
                                        )
                                    else:
                                        file = (
                                            "./communities/"
                                            + graph_model
                                            + "/K="
                                            + str(K)
                                            + "/"
                                            + met.split("_beta")[0]
                                            + "/beta="
                                            + met.split("_beta=")[1].split("_")[0]
                                            + "/communities_dict_"
                                            + met
                                            + "_num_iter_N_"
                                            + graph_model_full
                                            + "_rseed="
                                            + str(rseed)
                                            + "_a="
                                            + str(par)
                                        )

                                else:
                                    if "LFR" not in graph_model_full:
                                        file = (
                                            "./communities/"
                                            + graph_model
                                            + "/K="
                                            + str(K)
                                            + "/"
                                            + met.split("_beta")[0]
                                            + "/communities_dict_"
                                            + met
                                            + "_num_iter_N_"
                                            + graph_model_full
                                            + "_avg_degree="
                                            + str(avg_degree)
                                            + "_rseed="
                                            + str(rseed)
                                            + "_a="
                                            + str(par)
                                        )
                                    else:
                                        file = (
                                            "./communities/"
                                            + graph_model
                                            + "/K="
                                            + str(K)
                                            + "/"
                                            + met.split("_beta")[0]
                                            + "/communities_dict_"
                                            + met
                                            + "_num_iter_N_"
                                            + graph_model_full
                                            + "_rseed="
                                            + str(rseed)
                                            + "_a="
                                            + str(par)
                                        )

                                best_iteration, _ = automatic_best_iter(
                                    G,
                                    graph_model,
                                    file,
                                    max_iter=max_iter,
                                    metric=metric,
                                    path2save="./communities/" + met + "/",
                                    verbose=verbose,
                                )

                            else:
                                if "LFR" in graph_model:

                                    file = (
                                        "./communities/"
                                        + graph_model
                                        + "/K="
                                        + str(K)
                                        + "/"
                                        + met.split("_beta")[0]
                                        + "/communities_dict_"
                                        + met
                                        + "_"
                                        + graph_model_full
                                        + "_rseed="
                                        + str(rseed)
                                        + "_a="
                                        + str(par)
                                    )
                                else:

                                    file = (
                                        "./communities/"
                                        + graph_model
                                        + "/K="
                                        + str(K)
                                        + "/"
                                        + met.split("_beta")[0]
                                        + "/communities_dict_"
                                        + met
                                        + "_"
                                        + graph_model_full
                                        + "_avg_degree="
                                        + str(avg_degree)
                                        + "_rseed="
                                        + str(rseed)
                                        + "_a="
                                        + str(par)
                                    )

                                # path2met = './communities/'+graph_model+'/K='+str(K)+'/'+met.split('_beta')[0]+'/communities_dict_'+met+'_num_iter_'+str(num_iter)+'_'+graph_model_full+'_rseed='+str(rseed)+'_a='+str(par)
                                # all_files = os.listdir(path2met)
                                # file = all_files[0]

                                best_iteration = 1  # dummy

                            if verbose:
                                print("loading from file:", file)

                            if best_iteration != -1:

                                # read_outputs
                                file = file.replace(
                                    "iter_N_", "iter_" + str(best_iteration) + "_"
                                )

                                if "pkl" not in file:
                                    file += ".pkl"
                                if "/communities/" not in file:
                                    file = path2met + "/" + file

                                with open(file, "rb") as fp:
                                    g, community_mapping, _ = pkl.load(fp)

                                # compute metric for every output
                                try:
                                    communities = dict(G.nodes(data=comm_flag))
                                    com_count_dict = Counter(dict(communities).values())
                                    avg_com_size = sum(
                                        list(com_count_dict.values())
                                    ) / len(com_count_dict.keys())
                                    avg_com_size = int(avg_com_size)

                                    if verbose:
                                        print("GT number of nodes:", len(communities))
                                    if verbose:
                                        print(
                                            "inferred number of nodes:",
                                            len(community_mapping),
                                        )
                                    if verbose:
                                        print(
                                            "communities:",
                                            list(communities.values())[:6],
                                        )
                                    if verbose:
                                        print(
                                            "inferred com:",
                                            list(community_mapping.values())[:6],
                                        )

                                    if "LFR" in graph_model_full:

                                        communities = relabel_comm(communities)
                                        community_mapping = relabel_comm(
                                            community_mapping
                                        )

                                    community_mapping_new = utils.comm_permutation(
                                        dict(communities), community_mapping
                                    )

                                    if verbose:
                                        print(
                                            "communities:",
                                            list(dict(communities).values())[:6],
                                        )
                                    if verbose:
                                        print(
                                            "inferred relabeled com:",
                                            list(community_mapping.values())[:6],
                                        )
                                    if verbose:
                                        print(
                                            "inferred permuted com:",
                                            list(community_mapping_new.values())[:6],
                                        )

                                    value = utils.metric_comp(
                                        G,
                                        community_mapping_new,
                                        metric,
                                        clustering_label=(comm_flag),
                                    )
                                except:
                                    if verbose:
                                        print("breaking at permutation level")
                                    value = -1
                            else:

                                value = -1

                            data.append(
                                (met, par, avg_degree, eff_avg_degree, K, N, value)
                            )
                            best_iter_list.append(best_iteration)
                            rseed_full_list.append(rseed)
                            avg_degree_full_list.append(avg_degree)
                            eff_avg_degree_list.append(eff_avg_degree)
                            # avg_com_size_list.append(avg_com_size)
                            K_full_list.append(K)
                            N_full_list.append(N)

    df = pd.DataFrame(
        data,
        columns=["method", "param", "avg_degree", "eff_avg_deg", "K", "N"] + [metric],
    )
    df["best_iter"] = best_iter_list
    df["rseed"] = rseed_full_list
    df["avg_degree"] = avg_degree_full_list
    df["eff_avg_deg"] = eff_avg_degree_list
    # df['avg_com_size'] = avg_com_size_list
    df["K"] = K_full_list
    df["N"] = N_full_list
    return df


def line_plots(
    df,
    graph_model_full,
    metric,  # df feature to show in the y axis; normally computed quality metric
    x="param",  # df feature to show in the x axis
    extra=None,  # extra information to be added to the name of the output png file
):

    """
    Script to get line plots from df.
    """
    sns_plot = sns.lineplot(
        data=df,
        x=x,
        y=metric,
        hue="method",
        style="method",
        linewidth=2.5,
        marker="o",
        markersize=7.0,
    )
    for _, s in sns_plot.spines.items():
        s.set_linewidth(3)
        s.set_color("k")

    fig = sns_plot.get_figure()
    if extra is None:
        extra = "_"
    fig.savefig(
        "./plots/" + graph_model_full + "_" + metric + "_" + str(x) + extra + ".png",
        bbox_inches="tight",
        dpi=300,
        format="png",
    )
    plt.show()


def bar_plots(df, graph_model_full, metric):
    # plt.figure(figsize=(25,20))

    sns_plot = sns.lineplot(
        data=df,
        x=df["method"],
        y=metric,
        hue="method",
        style="method",
        linewidth=2.5,
        marker="o",
        markersize=7.0,
    )
    for _, s in sns_plot.spines.items():
        s.set_linewidth(3)
        s.set_color("k")

    fig = sns_plot.get_figure()
    fig.savefig(
        "./plots/" + graph_model_full + "_" + metric + ".png",
        bbox_inches="tight",
        dpi=300,
        format="png",
    )
    plt.show()


def plot_side_by_side(
    original_G,  # given network
    communities,  # ground truth communities
    outputs,  # dictionary containing community mapping as value and method as key
    method_list,  # list of methods to plot
    metric,  # metric to show as a title; usually ARI or F1
    comm_label="community",  # community label
    wt=1,  # # width for the edges
    alpha=0.5,  # alpha for the nodes
    ns=100,  # node size
    extra="_",  # extra string for saving
    spring_seed=0,  # seed for the spring layout; if set to be None, then community layout is used.
    differences=False,  # flag to highlight differences
    cmap=plt.cm.Set1,  # colormap for the plots.
):
    # @plot_side
    """
    Function to plot side by side networks.
    """
    if spring_seed is not None:
        pos = nx.spring_layout(original_G, seed=spring_seed)
    else:
        pos = utils.community_layout(original_G, dict(communities))
    fig, ax = plt.subplots(
        figsize=(8 * len(method_list), 8), nrows=1, ncols=len(method_list)
    )

    i = -1
    for col in ax:
        i += 1
        method = method_list[i]
        labels_true = []
        predicted_labels = []
        print(method)
        if method == "ground_truth":
            nbr_comm = len(set(list(dict(original_G.nodes(data=comm_label)).values())))
            value = "-"
            colors = list(dict(communities).values())
            colors = [int(c) for c in colors]
        else:
            G_, community_mapping, _ = outputs[method]
            community_mapping_new = utils.comm_permutation(
                dict(communities), community_mapping
            )
            value = utils.metric_comp(
                original_G, community_mapping_new, metric, clustering_label=(comm_label)
            )
            value = round(value, 4)

            colors = [int(community_mapping[node]) for node in G_.nodes()]
            nbr_comm = len(set(colors))

        if not differences or method == "ground_truth":
            nx.draw(
                original_G,
                pos=pos,
                node_size=ns,
                edge_color="gray",
                with_labels=False,
                node_color=colors,
                ax=col,
                cmap=cmap,
                width=wt,
                alpha=alpha
                # node_color='lightgreen'
            )
        else:
            missed_list = []
            correct_list = []
            for node in community_mapping_new.keys():
                if community_mapping_new[node] != communities[node]:
                    missed_list.append(node)
                else:
                    correct_list.append(node)

                nx.draw(
                    original_G,
                    pos=pos,
                    edge_color="gray",
                    node_size=0,
                    ax=col,
                    width=wt,
                )

                nx.draw_networkx_nodes(
                    original_G,
                    pos=pos,
                    node_size=ns,
                    # with_labels = False,
                    nodelist=correct_list,
                    node_color="gray",
                    ax=col,
                    # cmap=plt.cm.Set1,
                    # width = wt,
                    alpha=alpha
                    # node_color='lightgreen'
                )
                nx.draw_networkx_nodes(
                    original_G,
                    pos=pos,
                    node_size=ns,
                    # with_labels = False,
                    nodelist=missed_list,
                    node_color="red",
                    ax=col,
                    # cmap=plt.cm.Set1,
                    # width = wt,
                    alpha=alpha
                    # node_color='lightgreen'
                )

        col.set_title(
            method.split("_")[0]
            + "/ncom:"
            + str(nbr_comm)
            + "/"
            + metric
            + "="
            + str(value),
            fontsize=16,
        )
    fig.tight_layout()
    fig.savefig(
        "./plots/side-by-side_nnets:"
        + str(len(method_list))
        + "_metric:"
        + metric
        + "_"
        + extra
        + ".png"
    )
    plt.show()


def set_squares_synt(G, G2, communities, community_mapping):

    ncomm = list(set(dict(communities).values()))
    colours_list = colours[:]
    #####
    comm_orig, comm_met = communities2dict(communities, community_mapping)

    intersections = {}
    for i, j in comm_met.items():
        len_intersection = []
        for m, n in comm_orig.items():
            len_intersection.append((m, len(set(j).intersection(n))))
            # print(i,m,set(j).intersection(n))

        len_intersection.sort(key=lambda x: x[1])
        intersections[i] = len_intersection[-1]
    same_colour = []
    for i in intersections.keys():
        same_colour += set(comm_orig[intersections[i][0]]).intersection(comm_met[i])

    different_com = {}
    for idx, k in enumerate(comm_orig.keys()):
        for i in intersections.keys():
            if k != intersections[i][0]:
                # print(k,len(colours),idx+len(ncomm)+1)
                different_com[k] = colours[idx + 1]

    node_colours_inferred = {}
    for node in G2.nodes():
        if node in same_colour:
            node_colours_inferred[node] = nx.get_node_attributes(G, "colour")[node]
            G2.nodes[node]["marker"] = "o"
        else:

            node_colours_inferred[node] = different_com[community_mapping[node]]
            G2.nodes[node]["marker"] = "s"

    return node_colours_inferred, G2


def markers_for_real_nets_infomap(G, G2, communities, community_mapping):

    ncomm = list(set(dict(communities).values()))
    colours_list = colours[:]
    #####
    comm_orig, comm_met = communities2dict(communities, community_mapping)

    intersections = {}
    colors_of_original = {}
    for i, j in comm_met.items():
        len_intersection = []
        for m, n in comm_orig.items():
            len_intersection.append(
                (m, len(set(j).intersection(n)), len(set(j).intersection(n)) / len(n))
            )
            # print(i,m, n, j)

        len_intersection.sort(key=lambda x: x[1])
        intersections[i] = len_intersection[-1]
        node_in_c = comm_orig[len_intersection[-1][0]][0]
        colors_of_original[i] = nx.get_node_attributes(G, "colour")[node_in_c]
    # same_colour = []
    same_shape = []
    for i in intersections.keys():
        # print(set(comm_orig[intersections[i][0]]).intersection(comm_met[i]))
        # same_colour += set(comm_met[i]).intersection(comm_orig[intersections[i][0]])
        same_shape += set(comm_orig[intersections[i][0]]).intersection(comm_met[i])

    different_com = {}
    for idx, k in enumerate(comm_orig.keys()):
        for i in intersections.keys():
            if k != intersections[i][0]:
                # print(k,len(colours),idx+len(ncomm)+1)
                different_com[k] = colours[idx + 1]

    node_colours_inferred = {}
    for node in G2.nodes():
        col = colors_of_original[community_mapping[node]]
        node_colours_inferred[node] = col  # nx.get_node_attributes(G, "colour")[node]
        G2.nodes[node]["colour"] = col
        if node in same_shape:
            G2.nodes[node]["marker"] = "o"
        else:
            G2.nodes[node]["marker"] = "s"

        # else:

        #    node_colours_inferred[node] = different_com[community_mapping[node]]
        #    G2.nodes[node]["marker"] = "s"
    return node_colours_inferred, G2.copy()


def markers_for_real_nets(G, G2, communities, community_mapping):

    ncomm = list(set(dict(communities).values()))
    colours_list = colours[:]
    #####
    comm_orig, comm_met = communities2dict(communities, community_mapping)

    intersections = {}
    for i, j in comm_orig.items():
        len_intersection = []
        for m, n in comm_met.items():
            len_intersection.append(
                (m, len(set(j).intersection(n)), len(set(j).intersection(n)) / len(n))
            )
            # print(i,m, n, j)

        len_intersection.sort(key=lambda x: x[1])
        intersections[i] = len_intersection[-1]
    same_colour = []
    for i in intersections.keys():
        # print(set(comm_met[intersections[i][0]]).intersection(comm_orig[i]))
        same_colour += set(comm_met[intersections[i][0]]).intersection(comm_orig[i])

    different_com = {}
    for idx, k in enumerate(comm_met.keys()):
        for i in intersections.keys():
            if k != intersections[i][0]:
                # print(k,len(colours),idx+len(ncomm)+1)
                different_com[k] = colours[idx + len(ncomm) + 1]

    node_colours_inferred = {}
    for node in G2.nodes():
        if node in same_colour:
            node_colours_inferred[node] = nx.get_node_attributes(G, "colour")[node]
            G2.nodes[node]["marker"] = "o"
        else:

            node_colours_inferred[node] = different_com[community_mapping[node]]
            G2.nodes[node]["marker"] = "s"

    return node_colours_inferred, G2.copy()
