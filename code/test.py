import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys
import os
from tqdm import tqdm
import requests
import json

import main


url = "http://bost.ocks.org/mike/miserables/miserables.json"

lesmis = json.loads(requests.get(url).text)
G = nx.readwrite.json_graph.node_link_graph(lesmis, multigraph=False)
nodes = G.nodes()
mapping = {n:idx_+1 for idx_,n in enumerate(nodes)}
G = nx.relabel_nodes(G, mapping)


print(nx.info(G))
K = len(set(list(dict(G.nodes(data='group')).values())))
print('Number of communities:',K)
nx.draw(G, node_color = list(dict(G.nodes(data='group')).values()))
for node in G.nodes():G.nodes[node]['community'] = G.nodes[node]['group']
graph_model = 'les-mis'
met = 'Nextrout'
num_iter = 3
beta = 1

#### Running community detection #####

g,community_mapping,_ = main.community_detection(G, 
                                                   graph_model, 
                                                   met, 
                                                   num_iter, 
                                                   beta,#<----
                                                   plotting = False, 
                                                   path2save = './../data/',
                                                    K=K
                                                       )