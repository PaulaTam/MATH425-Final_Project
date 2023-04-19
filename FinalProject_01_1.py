#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:36:20 2023

@author: PaulaTam
"""
import matplotlib.pyplot as plt
import networkx as nx

from FinalProject_01 import nested_dict

G = nx.Graph()

def add_edges(dict_of_actors):
    for k1, k2 in dict_of_actors.items():
        for v in k2:
            print(k1, k2, k2[v])
            #G.add_edge(k1, k2, weight= k2[v])

add_edges(nested_dict)

pos = nx.spring_layout(G, seed=1)

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, width=6)
nx.draw_networkx_edges(
    G, pos, width=6, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()
#print(actor_ranks) 