import re
import networkx as nx
import numpy as np

def get_Jazz_network():
    with open('datasets\\Jazz-Musicians-Network.wl', 'r') as file:
        data = file.read()

    nodes_match = re.search(r'\{(.+?)\}', data)
    edges_match = re.findall(r'UndirectedEdge\[(\d+),\s*(\d+)\]', data)

    if nodes_match and edges_match:
        nodes_str = nodes_match.group(1)

        nodes = list(map(int, re.findall(r'\d+', nodes_str)))

        edges = [tuple(map(int, edge)) for edge in edges_match]

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        adjacency_matrix = nx.to_numpy_matrix(G, nodelist=nodes)

        return nodes, edges, adjacency_matrix
    else:
        print("no valid informaiton.")

