#import matplotlib.pyplot as plt
import networkx as nx

from FinalProject_01 import node_list

#creating directed network object for the actors where node1 points to node2
G = nx.DiGraph()
G.add_weighted_edges_from(node_list)
#https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.add_weighted_edges_from.html
#add_weighted_edges_from takes in a list of tuples
#e.g. [(node1, node2, weight)]

#Page Rank
page_rank = nx.pagerank(G)
#sorted pagerank in descending order
ordered_page_rank = sorted([(node, pagerank) for node, pagerank in page_rank.items()], key=lambda x:page_rank[x[0]], reverse=True)
print(ordered_page_rank[:10])

#commented out is the function initially used to try and make a visualization for the directed network
#however due to complications, the graph does not finish drawing in a reasonable time
"""
def save_network_graph(graph, file_name):
    #init figure
    plt.figure(figsize=(25, 25))
    plt.axis('off')
    
    pos = nx.spring_layout(G, k=3)
    weight = nx.get_edge_attributes(G,'weight')
    
    nx.draw_networkx_nodes(G, pos=pos, node_size=150)
    nx.draw_networkx_labels(G, pos=pos, font_size=8)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weight, font_size=6)
    
    plt.savefig(file_name, bbox_inches="tight", format="PDF")
    plt.show()
    
save_network_graph(G, "Problem_1_graph.pdf")
"""
