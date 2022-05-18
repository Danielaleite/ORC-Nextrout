import pickle as pkl
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
import networkx as nx

import plotting_tools as plotting

sys.path.append("../nextrout_tutorial/")
import utils

sys.path.append("./networks/synthetic_data_SBM/")
import SBM_generation as sbm
import os

from matplotlib.lines import Line2D


def get_label(graph_model):
    if "dol" in graph_model:
        label = "sex"
    elif "les-mis" in graph_model or "lesmis" in graph_model:
        label = "group"
    elif "foot" in graph_model:
        label = "value"
    elif "SBM" in graph_model or "sbm" in graph_model:
        label = "block"
    elif "book" in graph_model or "blogs" in graph_model:
        label = "value"
    elif "LFR" in graph_model:
        label = "community"
    else:
        print("wrong graph model!")
    return label


# def automatic_best_iter(
#     G, graph_model, results_path, max_iter, metric="ari", path2save="./data/"
# ):
#     iter2metric = {}
#     label = get_label(graph_model)
#     for itnum in range(1, max_iter + 1):
#         try:

#             new_path = results_path.replace("_num_iter_N", "_num_iter_" + str(itnum))

#             results = open(new_path + ".pkl", "rb")

#             [g, community_mapping, cutoff] = pkl.load(results)

#             value = utils.metric_comp(
#                 G, community_mapping, metric, clustering_label=(label)
#             )

#             iter2metric[itnum] = value
#         except:
#             pass
#     # print(iter2metric)
#     best_iter = max(iter2metric, key=iter2metric.get)
#     return best_iter, iter2metric[best_iter]


def relabel_comm(com_dict):
    com_dict = dict(com_dict)
    comm_list = list(com_dict.values())
    label2int = {}
    comm_list = list(set(comm_list))
    comm_list.sort()
    for idx, lab in enumerate(comm_list):
        label2int[lab] = idx
    com_dict_new = {}
    for node in com_dict.keys():
        com_dict_new[node] = label2int[com_dict[node]]
    return com_dict_new


def get_net(graph_flag, relabel=True):

    if graph_flag == "les-mis":
        import requests
        import json

        url = "http://bost.ocks.org/mike/miserables/miserables.json"

        lesmis = json.loads(requests.get(url).text)
        G = nx.readwrite.json_graph.node_link_graph(lesmis, multigraph=False)

    elif graph_flag == "dolphins":
        path = "../../data/real_nets/dolphins/dolphins2.gml"
        G = nx.read_gml(path)

    elif graph_flag == "american-football":
        path = "../../data/real_nets/americanFootball/football.gml"
        G = nx.read_gml(path)

    elif graph_flag == "political-blogs":
        path = "../../data/real_nets/polblogs/polblogs.gml"
        G = nx.read_gml(path)
        G = nx.Graph(G)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])

    elif graph_flag == "political-books":
        path = "../../data/real_nets/polbooks/polbooks.gml"
        G = nx.read_gml(path)

    else:
        print("error")

    if relabel:
        # relabeling
        real2nx_dict = {}
        for idx, node in enumerate(G.nodes()):
            real2nx_dict[node] = idx
        G = nx.relabel_nodes(G, real2nx_dict)
    return G


def df_for_line_plots_real(
    graph_model, graph_model_full, graph_flags, comm_flags, method_list, metric
):

    data = []
    best_iter_list = []

    for met in method_list:  # tqdm(method_list)

        for i, gf in enumerate(graph_flags):

            comm_flag = comm_flags[i]

            G = get_net(gf)

            if "Mix" in met or "Next" in met or "McOpt" in met:

                file = (
                    "./communities/"
                    + gf
                    + "/"
                    + met.split("_beta")[0]
                    + "/communities_dict_"
                    + met
                    + "_num_iter_N_"
                    + graph_model_full
                )

                best_iteration, _ = automatic_best_iter(
                    G,
                    gf,
                    file,
                    max_iter=15,
                    metric=metric,
                    path2save="./communities/" + met + "/",
                )

            else:
                all_files = os.listdir(
                    "./communities/" + gf + "/" + met.split("_beta")[0]
                )
                file = all_files[0]
                best_iteration = file.split("iter_")[1].split("_")[0]

            # read_outputs

            file = (
                "./communities/"
                + gf
                + "/"
                + met.split("_beta")[0]
                + "/communities_dict_"
                + met
                + "_num_iter_"
                + str(best_iteration)
                + "_"
                + graph_model_full
                + ".pkl"
            )

            with open(file, "rb") as fp:
                g, community_mapping, _ = pkl.load(fp)

            communities = G.nodes(data=comm_flag)

            communities = relabel_comm(communities)

            G = plotting.set_markers_gt(G, communities, gt=True)

            community_mapping_new = utils.comm_permutation(
                dict(communities), community_mapping
            )
            if (
                False
            ):  # set to True if single node communities should be added into a single comm
                _, isolated_nodes = plotting.node_colouring(
                    G, g, communities, community_mapping, ndiff=3
                )
                community_mapping_isolated = community_mapping.copy()
                last_comm = community_mapping[isolated_nodes[0][0]]
                for isol in isolated_nodes:
                    community_mapping_isolated[isol[0]] = last_comm
            value = utils.metric_comp(
                G, community_mapping_isolated, metric, clustering_label=(comm_flag)
            )

            data.append((met, gf, value))
            best_iter_list.append(best_iteration)

    df = pd.DataFrame(data, columns=["method", "graph_flag"] + [metric])

    ### Plotting ###

    df = pd.DataFrame(data, columns=["method", "dataset"] + [metric])
    df["best_iter"] = best_iter_list
    return df


def line_plots(df, graph_model_full, metric):
    # plt.figure(figsize=(25,20))

    sns_plot = sns.lineplot(
        data=df,
        x="param",
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
    original_G,
    communities,
    outputs,
    method_list,
    metric,
    cm,
    comm_label="community",
    wt=1,
    alpha=0.5,
    ns=50,
):

    pos = nx.spring_layout(original_G, seed=12)
    fig, ax = plt.subplots(
        figsize=(8 * len(method_list), 8), nrows=1, ncols=len(method_list)
    )

    i = -1
    for col in ax:
        i += 1
        method = method_list[i]
        labels_true = []
        predicted_labels = []
        if method == "ground_truth":
            nbr_comm = len(set(list(dict(original_G.nodes(data=comm_label)).values())))
            value = "-"
            colors = list(dict(communities).values())
            colors = [int(c) for c in colors]
        else:
            G_, community_mapping, _ = outputs[method]

            communities = original_G.nodes(data=comm_label)

            communities = relabel_comm(communities)

            new_community_mapping = utils.comm_permutation(
                dict(communities), community_mapping
            )

            colors = [int(community_mapping[node]) for node in G_.nodes()]

            nbr_comm = len(set(colors))
            value = utils.metric_comp(
                original_G, community_mapping, metric, clustering_label=comm_label
            )
            value = round(value, 3)
        nx.draw(
            original_G,
            pos=pos,
            node_size=ns,
            edge_color="gray",
            with_labels=False,
            # node_color = colors,
            ax=col,
            node_color=[cm(colors[i]) for i, n in enumerate(original_G.nodes())],
            cmap=plt.cm.Set1,
            width=wt,
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
        + ".png"
    )
    plt.show()


def automatic_best_iter(
    G,
    graph_model,
    results_path,
    max_iter,
    metric="ari",
    path2save="./data/",
    verbose=False,
):
    # @automatic
    iter2metric = {}
    if "dol" in graph_model:
        label = "sex"
    elif "les-mis" in graph_model or "lesmis" in graph_model:
        label = "group"
    elif "foot" in graph_model:
        label = "value"
    elif "SBM" in graph_model or "sbm" in graph_model:
        label = "block"
    elif "book" in graph_model:
        label = "value"
    elif "LFR" in graph_model:
        label = "community"
    else:
        if verbose:
            print("wrong graph model!")
    for itnum in range(1, max_iter + 1):
        try:
            # print('worked!')
            new_path = results_path.replace("_num_iter_N", "_num_iter_" + str(itnum))
            # print('new',new_path)
            results = open(new_path + ".pkl", "rb")
            try:
                # print('worked (2)!')
                [g, community_mapping, cutoff] = pkl.load(results)

                value = utils.metric_comp(
                    G, community_mapping, metric, clustering_label=(label)
                )

                iter2metric[itnum] = value
            except:
                if verbose:
                    print("breaking at loading level:" + results_path, " iter:", itnum)
        except:
            pass
    # print(iter2metric)

    try:
        # print('worked!')
        best_iter = max(iter2metric, key=iter2metric.get)
    except:
        best_iter = -1
        iter2metric[best_iter] = -1
        if verbose:
            print("missing: " + results_path)
    return best_iter, iter2metric[best_iter]


def data_counter(graph_model, N, K, avg_degree, comm_flag, expected_number):
    # @data_counter

    graph_model_full = (
        graph_model + "_N" + str(N) + "_K=" + str(K) + "_avg_degree=" + str(avg_degree)
    )

    subbeta_list = os.listdir(
        "./communities/" + graph_model + "/K=" + str(K) + "/Nextrout/"
    )
    subbeta_list.sort()
    print(subbeta_list)
    nextrout_files = []
    count = 0

    for beta in subbeta_list:
        nextrout_files += os.listdir(
            "./communities/" + graph_model + "/K=" + str(K) + "/Nextrout/" + beta
        )

    beta_count = {}
    beta_list = []
    param_list = []
    param_count = {}
    print("N={:},K={:},avg_degree={:}".format(N, K, avg_degree))
    for f in nextrout_files:
        if (
            "N" + str(N) in f
            and "K=" + str(K) in f
            and "avg_degree=" + str(avg_degree) in f
        ) or ("LFR_N500" in f):
            # beta-----------------------------------
            beta = f.split("_beta=")[1].split("_")[0]
            if beta not in beta_list:
                beta_list.append(beta)
                beta_count[beta] = 1
            else:
                beta_count[beta] += 1
            param = float(f.split("_a=")[1].split(".pkl")[0])
            if param not in param_list:
                param_list.append(param)
            try:
                param_count[beta] += [param]
            except:
                param_count[beta] = []
            count += 1
    print("number of files", count)
    param_list.sort()
    print("found betas:", beta_list)
    print("found parameters:", param_list)
    for key in param_count.keys():
        temp_list = list(set(param_count[key]))
        temp_list.sort()
        param_count[key] = temp_list
    print("found per beta:", param_count)

    data_percentage = []
    for key in beta_count.keys():

        betas_divided_param = round(beta_count[key] / len(param_list), 2)

        print(
            "beta: {:4}, nets per beta and param (rseeds x iters) {:6}, %: {:5}:".format(
                key,
                betas_divided_param,
                round(betas_divided_param * 100 / expected_number, 2),
            )
        )
        data_percentage.append(round(betas_divided_param * 100 / expected_number, 2))

    print(beta_count)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.bar(beta_count.keys(), data_percentage)
    ax.set_title(
        "N={:},K={:},avg_degree={:}, expected nbr of nets (seeds x iters)= {:}".format(
            N, K, avg_degree, expected_number
        )
    )
    return beta_list, param_list
