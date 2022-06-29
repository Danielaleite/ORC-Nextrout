import sys

sys.path.append("./plot_code/")
import vizCM as viz
import pandas as pd
import networkx as nx
import numpy as np
import os


def output_adjacency(
    G, comm_label, out_folder="./data/input/", verbose=False, outfile=None
):
    """
        Output the adjacency matrix. Default format is space-separated .csv with 3 columns:
        node1 node2 weight

        Parameters
        ----------
        G: Graph
           Graph NetworkX object.
        com_label: str
            communtiy label
        outfile: str
                 Name of the adjacency matrix.
    """
    # Getting the GT communities
    communities = nx.get_node_attributes(G, comm_label)

    # Defining empty lists to store node belonging to a certain community k
    groups = {
        int(k): [] for k in set(communities.values())
    }  # assuming sequential community labels
    for n in communities:
        # Appending them into the lists
        k = int(communities[n])
        groups[k].append(n)

    # Defining K
    # K = max(list(groups.keys()))
    K = len(list(groups.keys()))
    print("K inside output", K)
    # Writing edgelist to a file
    nx.write_edgelist(G, out_folder + outfile, data=False, delimiter=",")
    # Loading the file as df and adding extra "weight" column
    df = pd.read_csv(out_folder + outfile, header=None)
    df[2] = 1  # 2nd col = weight
    # Storing this into a csv
    df.to_csv(out_folder + outfile, index=False, header=False)

    # Writing membership information into files
    nodes = list(G.nodes())
    nodes.sort()  # sorting is important since some algorithms output sorted things
    N = G.number_of_nodes()
    # Defining memberships
    u = np.zeros((N, K)).astype("int")
    for i, n in enumerate(nodes):
        # print(i,n)
        k = (
            int(communities[n]) - 1
        )  # index and node label must be equal (and sorted)! this corrects it.
        u[i, k] = 1  # node i in community k then 1 for u
    assert np.all(
        np.sum(u, axis=1) == 1
    )  # checking that all the nodes belong to a community

    # Storing GT information to be used by MT
    outfile_theta = "theta_gt_" + outfile
    np.savez_compressed(out_folder + outfile_theta[: -len(".dat")], u=u, v=u)
    if verbose:
        print(f"Adjacency matrix saved in: {out_folder+ outfile}")


def preprocessing(
    G, comm_label, path2save="./../test_crep/data/input/", name="adj.dat", verbose=True
):
    output_adjacency(G, comm_label, out_folder=path2save, outfile=name, verbose=verbose)


def MT_comm_dect(
    G,
    K,
    comm_label,
    path2save="./../test_crep/data/input/",
    name="adj.dat",
    verbose=True,
):

    import sys

    # generating input files

    preprocessing(G, comm_label, path2save=path2save, name=name, verbose=verbose)

    # running MT
    current_dir = os.getcwd()
    os.chdir("./../test_crep/src/")
    os.system("python analyse_data.py -n " + name + " -k " + str(K))
    os.chdir(current_dir)

    # postprocessing results (u)
    print("using eta_nc as algorithm.")
    path2load = "./../test_crep/data/output/eta_nc/"
    theta_gt = np.load(
        path2save + "/theta_gt_" + name[: -len(".dat")] + ".npz"
    )  # ground truth info
    theta_inf = np.load(
        path2load + "/theta_" + name[: -len(".dat")] + ".npz"
    )  # inferred communities

    # float values for the memberships
    u_array_gt = theta_gt["u"]
    u_array_gt = viz.normalize_nonzero_membership(u_array_gt)
    u_gt = np.argmax(
        u_array_gt, axis=1
    )  # flattening u: we get the index s.t. the entry is equal to 1
    # if for i = 1 this column is 20, then u_gt has a 20 in the entry 1 (this is the community assignment).

    # Similar for GT
    u_array_inf = theta_inf["u"]
    u_norm_array = viz.normalize_nonzero_membership(u_array_inf)
    u_norm = np.argmax(u_norm_array, axis=1)

    return u_norm, u_gt
