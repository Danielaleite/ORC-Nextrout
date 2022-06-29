import yaml as yaml
import os, copy
import MTrep as MTREP
import bptf as BPTF
import csv
from argparse import ArgumentParser
import tools as tl
import igraph as ig
import numpy as np
import leidenalg
import networkx as nx

np.seterr(divide='ignore', invalid='ignore')


def main_analyse_data():

    p = ArgumentParser()
    p.add_argument('-a', '--algorithm', type=str, default='eta_nc')
    p.add_argument('-n', '--adj', type=str, default='adj_0.dat')
    p.add_argument('-k', '--K', type=int, default=2)
    p.add_argument('-f', '--infolder', type=str, default='../data/input/')

    args = p.parse_args()

    algorithm = args.algorithm
    infolder = args.infolder
    adj = args.adj
    K = args.K

    out_folder = '../data/output/'+algorithm
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if algorithm not in ['leiden']:
        configuration = 'setting_'+algorithm+'.yaml'
        with open(configuration) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        with open(out_folder + '/setting.yaml', 'w') as f:
            yaml.dump(conf, f)

        conf['end_file'] = '_'+adj.split('.dat')[0] + algorithm

    out_file = out_folder + '/results.csv'
    print('Output will be saved in ',out_file)
    if not os.path.isfile(out_file):  # write header
        with open(out_file, 'w') as outfile:
            wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
            wrtr.writerow(['network', 'K', 'rseed', 'CS', 'F1', 'JACCARD', 'NMI'])

    with open(out_file, 'a') as outfile:
        wrtr = csv.writer(outfile, delimiter=',', quotechar='"')

        A, B = tl.import_data(infolder + adj, ego=0, alter=1, force_dense=True, header=None,delimiter=None)
        if conf['undirected'] == True:
            for l in range(B.shape[0]):
                B[l] = np.triu(B[l]) + np.triu(B[l]).T
                assert np.allclose(B[l],np.tril(B[l]) + np.tril(B[l]).T)
                # B[l] = np.max(B[l],B[l].T) # make it symmetric

        nodes = A[0].nodes()
        theta = np.load(infolder + 'theta_gt_' + adj.replace('.dat', '.npz') )
        Ugt = theta['u']
        Vgt = theta['v']
        K = Ugt.shape[1]
        
        outfile_theta = out_folder + '/theta_' + adj.replace('.dat', '.npz')

        if algorithm == 'leiden':
            g2 = ig.Graph.Adjacency((nx.to_numpy_matrix(A[0]) > 0).tolist())
            part = leidenalg.find_partition(g2, leidenalg.ModularityVertexPartition, n_iterations=5)
            Kf = len(part)
            d = [list(el) for el in part]
            np.savez_compressed(outfile_theta, d=d)

            cs = np.nan
            f1 = tl.evalu(d, Ugt, 'f1', True)
            jc = tl.evalu(d, Ugt, 'jaccard', True)
            try:
                nmi = tl.nmiv(d, Ugt, True)[0]
            except:
                nmi = 0.

        else:
            Kf = K
            if algorithm == 'bptf':
                mod = BPTF.BPTF(n_modes=B.ndim, n_components=K, **conf)
                mod.fit(B)
                pars = mod.G_DK_M_f
                Wf, Uf, Vf = pars[0], pars[1], pars[2]
                np.savez_compressed(outfile_theta, u=Uf, v=Vf, w=Wf)

            else:
                mod = MTREP.MTrep(N=A[0].number_of_nodes(), L=len(A), K=K, **conf)
                Uf, Vf, Wf, nuf, maxLf = mod.fit(B, nodes)
                np.savez_compressed(outfile_theta, u=Uf, v=Vf, w=Wf, eta=nuf)

            
            Uf = tl.normalize_nonzero_membership(Uf)
            Vf = tl.normalize_nonzero_membership(Vf)
            
            Uf_perm,cs_u = tl.cosine_similarity(Uf, Ugt)
            Vf_perm,cs_v = tl.cosine_similarity(Vf, Vgt)
            cs = 0.5 * (cs_u+cs_v)
            print('cs',cs)
            f1 = (tl.evalu(Uf, Ugt, 'f1') + tl.evalu(Vf, Vgt, 'f1')) / 2.
            print('f1',f1)
            jc = (tl.evalu(Uf, Ugt, 'jaccard') + tl.evalu(Vf, Vgt, 'jaccard')) / 2.
            print(cs,f1,jc)
            import sklearn.metrics as sm
            ari1 = sm.adjusted_rand_score(np.argmax(Ugt, axis=1), np.argmax(Uf, axis=1) )     
            ari2 = sm.adjusted_rand_score(np.argmax(Vgt, axis=1), np.argmax(Vf, axis=1) )     
            try:
                nmi = (tl.nmiv(Uf, Ugt)[0] + tl.nmiv(Vf, Vgt)[0]) / 2.
            except:
                nmi = 0.


        rseed = adj.split('/syn')[-1]
        comparison = [algorithm, Kf, rseed, cs,f1, jc, nmi, ari1, ari2]
        wrtr.writerow(comparison)
        outfile.flush()
        print(comparison)


if __name__ == '__main__':
    main_analyse_data()