""" Functions for handling the data. """

import networkx as nx
import numpy as np
import pandas as pd
import itertools
from cdlib import evaluation


def import_data(dataset, undirected=False, ego='Ego', alter='Alter', force_dense=True, header=None,delimiter=None):
    """
        Import data, i.e. the adjacency tensor, from a given folder.

        Return the NetworkX graph and its numpy adjacency tensor.

        Parameters
        ----------
        dataset : str
                  Path of the input file.
        undirected : bool
                     If set to True, the algorithm considers an undirected graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        force_dense : bool
                      If set to True, the algorithm is forced to consider a dense adjacency tensor.
        header : int
                 Row number to use as the column names, and the start of the data.

        Returns
        -------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
        B : ndarray
            Graph adjacency tensor.
    """

    # read adjacency file
    if delimiter is None:
        df_adj = pd.read_csv(dataset, header=header)
    else:df_adj = pd.read_csv(dataset, sep=delimiter, header=header)
    print('{0} shape: {1}'.format(dataset, df_adj.shape))

    A = read_graph(df_adj=df_adj, ego=ego, alter=alter, undirected=undirected)
    # print_graph_stat(A)

    nodes = list(A[0].nodes())

    # save the multilayer network in a numpy tensor with all layers
    B = build_B_from_A(A, force_dense=force_dense, nodes=nodes)

    return A, B


def read_graph(df_adj, ego='Ego', alter='Alter', undirected=False, noselfloop=True):
    """
        Create the graph by adding edges and nodes.
        It assumes that columns of layers are from l+2 (included) onwards.

        Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.

        Parameters
        ----------
        df_adj : DataFrame
                 Pandas DataFrame object containing the edges of the graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        undirected : bool
                     If set to True, the algorithm considers an undirected graph.
        noselfloop : bool
                     If set to True, the algorithm removes the self-loops.

        Returns
        -------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    # build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = list(set(egoID).union(set(alterID)))
    nodes.sort()

    L = df_adj.shape[1] - 2  # number of layers
    # build the multilayer NetworkX graph: create a list of graphs, as many graphs as there are layers
    if undirected:
        A = [nx.MultiGraph() for _ in range(L)]
    else:
        A = [nx.MultiDiGraph() for _ in range(L)]
    # set the same set of nodes and order over all layers
    for l in range(L):
        A[l].add_nodes_from(nodes)

    for index, row in df_adj.iterrows():
        v1 = row[ego]
        v2 = row[alter]
        for l in range(L):
            if row[l + 2 ] > 0:
                if A[l].has_edge(v1, v2):
                    A[l][v1][v2][0]['weight'] += int(row[l + 2])   # if the edge existed already, no parallel edges created
                else:
                    A[l].add_edge(v1, v2, weight=int(row[l + 2]))

    # remove self-loops
    if noselfloop:
        for l in range(L):
            A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

    return A


def print_graph_stat(A):
    """
        Print the statistics of the graph A.

        Parameters
        ----------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    L = len(A)
    N = A[0].number_of_nodes()
    print('Number of nodes =', N)
    print('Number of layers =', L)

    print('Number of edges and average degree in each layer:')
    avg_edges = 0
    avg_density = 0
    avg_M = 0
    avg_densityW = 0
    for l in range(L):
        E = A[l].number_of_edges()
        k = 2 * float(E) / float(N)
        M = np.sum([d['weight'] for u, v, d in list(A[l].edges(data=True))])
        kW = 2 * float(M) / float(N )

        print('E[', l, '] =', E, " -  <k> =", np.round(k, 2))
        print('M[', l, '] =', M, " <k_weighted> =", kW)
        avg_edges += E
        avg_density += k
        avg_M += M
        avg_densityW += kW
    print('Average edges over all layers:', np.round(avg_edges / L, 3))
    print('Average degree over all layers:', np.round(avg_density / L, 2))
    print('Average edges over all layers (weighted):', np.round(avg_M / L, 3))
    print('Average degree over all layers (weighted):', np.round(avg_densityW / L, 2))
    print('Total number of edges:', avg_edges)
    print('Total number of edges (weighted):', avg_M)


def build_B_from_A(A, force_dense=True, nodes=None):

    N = A[0].number_of_nodes()
    if nodes is None:
        nodes=list(A[0].nodes())
    B = np.empty(shape=[len(A), N, N],dtype='int')
    for l in range(len(A)):
        if force_dense:
            B[l, :, :] = np.array(nx.to_numpy_matrix(A[l], weight='weight', dtype=int, nodelist=nodes))
        else:
            try:
                B[l, :, :] = nx.to_scipy_sparse_matrix(A[l], dtype=int, nodelist=nodes)
            except:
                print('Warning: layer ', l, ' cannot be made sparse, using a dense matrix for it.')
                B[l, :, :] = np.array(nx.to_numpy_matrix(A[l], weight='weight', dtype=int, nodelist=nodes))

    return B


def reciprocal_edges(G):
    """
        Compute the proportion of bi-directional edges, by considering the unordered pairs.

        Parameters
        ----------
        G: MultiDigraph
           MultiDiGraph NetworkX object.

        Returns
        -------
        reciprocity: float
                     Reciprocity value, intended as the proportion of bi-directional edges over the unordered pairs.
    """

    n_all_edge = G.number_of_edges()
    # unique pairs of edges, i.e. number of edge in the undirected graph
    n_undirected = G.to_undirected().number_of_edges()
    # number of undirected edges that are reciprocated in the directed network
    n_overlap_edge = (n_all_edge - n_undirected)

    if n_all_edge == 0:
        raise nx.NetworkXError("Not defined for empty graphs.")

    reciprocity = float(n_overlap_edge) / float(n_undirected)

    return reciprocity


def can_cast(string):
    """
        Verify if one object can be converted to integer object.

        Parameters
        ----------
        string : int or float or str
                 Name of the node.

        Returns
        -------
        bool : bool
               If True, the input can be converted to integer object.
    """

    try:
        int(string)
        return True
    except ValueError:
        return False


def normalize_nonzero_membership(U):
    """
        Given a matrix, it returns the same matrix normalized by row.

        Parameters
        ----------
        U: ndarray
           Numpy Matrix.

        Returns
        -------
        The matrix normalized by row.
    """

    den1 = U.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return U / den1


def evalu(U_infer, U0, metric='f1', com=False):
    """
        Compute an evaluation metric.

        Compare a set of ground-truth communities to a set of detected communities. It matches every detected
        community with its most similar ground-truth community and given this matching, it computes the performance;
        then every ground-truth community is matched with a detected community and again computed the performance.
        The final performance is the average of these two metrics.

        Parameters
        ----------
        U_infer : ndarray
                  Inferred membership matrix (detected communities).
        U0 : ndarray
             Ground-truth membership matrix (ground-truth communities).
        metric : str
                 Similarity measure between the true community and the detected one. If 'f1', it used the F1-score,
                 if 'jaccard', it uses the Jaccard similarity.
        com : bool
              Flag to indicate if U_infer contains the communities (True) or if they have to be inferred from the
              membership matrix (False).

        Returns
        -------
        Evaluation metric.
    """

    if metric not in {'f1', 'jaccard'}:
        raise ValueError('The similarity measure can be either "f1" to use the F1-score, or "jaccard" to use the '
                         'Jaccard similarity!')

    K = U0.shape[1]

    gt = {}
    d = {}
    threshold = 1 / U0.shape[1]
    for i in range(K):
        gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
        if com:
            try:
                d[i] = U_infer[i]
            except:
                pass
        else:
            d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())
    # First term
    R = 0
    for i in np.arange(K):
        ground_truth = set(gt[i])
        _max = -1
        M = 0
        for j in d.keys():
            detected = set(d[j])
            if len(ground_truth & detected) != 0:
                precision = len(ground_truth & detected) / len(detected)
                recall = len(ground_truth & detected) / len(ground_truth)
                if metric == 'f1':
                    M = 2 * (precision * recall) / (precision + recall)
                elif metric == 'jaccard':
                    M = len(ground_truth & detected) / len(ground_truth.union(detected))
            if M > _max:
                _max = M
        R += _max
    # Second term
    S = 0
    for j in d.keys():
        detected = set(d[j])
        _max = -1
        M = 0
        for i in np.arange(K):
            ground_truth = set(gt[i])
            if len(ground_truth & detected) != 0:
                precision = len(ground_truth & detected) / len(detected)
                recall = len(ground_truth & detected) / len(ground_truth)
                if metric == 'f1':
                    M = 2 * (precision * recall) / (precision + recall)
                elif metric == 'jaccard':
                    M = len(ground_truth & detected) / len(ground_truth.union(detected))
            if M > _max:
                _max = M
        S += _max

    return np.round(R / (2 * len(gt)) + S / (2 * len(d)), 4)


# def cosine_similarity(U_infer, U0):
#     """
#         Compute the cosine similarity between ground-truth communities and detected communities.

#         Parameters
#         ----------
#         U_infer : ndarray
#                   Inferred membership matrix (detected communities).
#         U0 : ndarray
#              Ground-truth membership matrix (ground-truth communities).

#         Returns
#         -------
#         RES : float
#               Cosine similarity value.
#     """

#     N, C = U0.shape
#     # compute all possible permutations
#     lst = list(itertools.permutations(np.arange(C)))
#     RES = np.zeros((len(lst), 1))
#     for idx, el in enumerate(lst):
#         cos_sim = np.diag(np.dot(U_infer[:, el], U0.T))/(np.linalg.norm(U_infer[:, el], axis=1)*np.linalg.norm(U0, axis=1))
#         cos_sim[np.isnan(cos_sim)] = 0.
#         RES[idx] = np.sum(cos_sim)/N

#     return np.max(RES)

def CalculatePermuation(U_infer,U0):  
    """
    Permuting the overlap matrix so that the groups from the two partitions correspond
    U0 has dimension NxK, reference memebership
    """
    N,RANK=U0.shape
    M=np.dot(np.transpose(U_infer),U0)/float(N);   #  dim=RANKxRANK
    rows=np.zeros(RANK);
    columns=np.zeros(RANK);
    P=np.zeros((RANK,RANK));  # Permutation matrix
    for t in range(RANK):
    # Find the max element in the remaining submatrix,
    # the one with rows and columns removed from previous iterations
        max_entry=0.;c_index=0;r_index=0;
        for i in range(RANK):
            if columns[i]==0:
                for j in range(RANK):
                    if rows[j]==0:
                        if M[j,i]>max_entry:
                            max_entry=M[j,i];
                            c_index=i;
                            r_index=j;
     
        P[r_index,c_index]=1;
        columns[c_index]=1;
        rows[r_index]=1;

    return P


def cosine_similarity(U_infer,U0):
    """
    I'm assuming row-normalized matrices 
    """
    P=CalculatePermuation(U_infer,U0) 
    U_infer=np.dot(U_infer,P);      # Permute infered matrix
    N,K=U0.shape
    U_infer0=U_infer.copy()
    U0tmp=U0.copy()
    cosine_sim=0.
    norm_inf=np.linalg.norm(U_infer,axis=1)
    norm0=np.linalg.norm(U0,axis=1  )
    for i in range(N):
        if(norm_inf[i]>0.):U_infer[i,:]=U_infer[i,:]/norm_inf[i]
        if(norm0[i]>0.): U0[i,:]=U0[i,:]/norm0[i]
       
    for k in range(K):
        cosine_sim+=np.dot(np.transpose(U_infer[:,k]),U0[:,k])
    U0=U0tmp.copy()
    return U_infer0,cosine_sim/float(N) 
    



def nmiv(U_infer, U0, com=False):
    """
        Compute the normalized mutual information between ground-truth communities and detected communities.

        It uses the overlapping_normalized_mutual_information_LFK from cdlib library.

        Parameters
        ----------
        U_infer : ndarray
                  Inferred membership matrix (detected communities).
        U0 : ndarray
             Ground-truth membership matrix (ground-truth communities).

        Returns
        -------
        overlapping_normalized_mutual_information_LFK.
    """

    gt = []
    d = []
    threshold = 1/U0.shape[1]
    for i in range(U0.shape[1]):
        gt.append(list(np.argwhere(U0[:, i] > threshold).flatten()))
        if com:
            try:
                d.append(U_infer[i])
            except:
                pass
        else:
            d.append(list(np.argwhere(U_infer[:, i] > threshold).flatten()))

    return evaluation.overlapping_normalized_mutual_information_LFK(d, gt)


