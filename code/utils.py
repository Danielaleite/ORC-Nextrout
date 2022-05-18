def data2cdlib(graph, threshold, beta_d=1.0, MaxNumIter=5):  # utils
    ncom, g_communities = find_communities(graph, threshold, beta_d=1.0, MaxNumIter=5)
    ncom = nx.number_connected_components(g_communities)

    communities_subgraphs = [
        g_communities.subgraph(cc).copy()
        for cc in sorted(nx.connected_components(g_communities), key=len, reverse=True)
    ]

    com_list = [list(i.nodes()) for i in communities_subgraphs]
    edge_comm = [list(i.edges()) for i in communities_subgraphs]
    final_coms = NodeClustering(com_list, graph=None, method_name="nextrout")

    return final_coms


def methods_comparison(graph, method):  # utils

    import igraph as ig

    graph_ig = ig.Graph.from_networkx(graph)

    if method == "spinglass":
        dendrogram = graph_ig.community_spinglass()
    elif method == "infomap":
        dendrogram = graph_ig.community_infomap()
    elif method == "label_propagation":
        dendrogram = graph_ig.community_label_propagation()
    elif method == "edge_betweenness":
        dendrogram = graph_ig.community_edge_betweenness()
        # convert it into a flat clustering
        dendrogram = dendrogram.as_clustering()
    else:
        assert False, "method not found"
    # get the membership vector
    membership = dendrogram.membership
    # get dictionary
    community_mapping = {}
    for idx, node in enumerate(graph.nodes()):
        community_mapping[node] = membership[idx]
    return community_mapping


def generate_ground_truth(Graph, clustering_label="block"):  # utils
    """
    Generates the ground truth NodeClustering object based on a clustering label.

    Parameters
    ----------
    G : NetworkX graph
        the NetworkX graph used in input.
    clustering_label : str
        Node attribute name for ground truth.
    """

    att_list = nx.get_node_attributes(Graph, clustering_label)
    by_value = operator.itemgetter(1)
    val_list = [
        list(dict(g).keys())
        for k, g in groupby(sorted(att_list.items(), key=by_value), by_value)
    ]

    gt_comm = NodeClustering(val_list, Graph, method_name=None)

    return gt_comm


def get_label(graph_model):

    import utils

    print("get_label", graph_model)
    # if 'dol' in graph_model:
    #    label = 'sex'
    # elif 'les-mis' in graph_model or 'lesmis' in graph_model:
    #    label = 'group'
    # elif 'foot' in graph_model:
    #    label = 'value'
    if "SBM" in graph_model or "sbm" in graph_model:
        label = "block"
    # elif 'book' in graph_model or 'blogs' in graph_model:
    #    label = 'value'
    elif "real" in graph_model:
        label = "community"
    elif "LFR" in graph_model:
        label = "community"
    elif "complete" in graph_model:
        label = "community"
    else:
        print("wrong graph model!")
    return label
