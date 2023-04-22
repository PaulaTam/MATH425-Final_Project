#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:36:20 2023

@author: PaulaTam
"""
import matplotlib.pyplot as plt
import networkx as nx

from FinalProject_01 import node_list

G = nx.DiGraph()
G.add_weighted_edges_from(node_list)
weight = nx.get_edge_attributes(G,'weight')

def save_network_graph(graph, file_name):
    #init figure
    plt.figure(figsize=(25, 25))
    plt.axis('off')
    
    pos = nx.spring_layout(G, k=3)
    
    nx.draw_networkx_nodes(G, pos=pos, node_size=150)
    nx.draw_networkx_labels(G, pos=pos, font_size=8)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weight, font_size=6)
    
    plt.savefig(file_name, bbox_inches="tight", format="PDF")
    plt.show()
    
save_network_graph(G, "Problem_1_graph.pdf")