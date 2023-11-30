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

        adj_matrix = nx.to_numpy_matrix(G, nodelist=nodes)

        return nodes, edges, adj_matrix
    else:
        print("no valid informaiton.")

def get_USAir_network():
    edges_content = []
    with open('datasets\\US-Air-Lines.wl', 'r') as file:
        in_edges_section = False
        for line in file:
            if line.startswith('*Edges'):
                in_edges_section = True
            elif in_edges_section:
                edges_content.append(line.strip())
    file.close()

    num_vertices = 332

    direct_adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    for line in edges_content:
        vertex1, vertex2, weight = [float(val) for val in line.split()]
        direct_adj_matrix[int(vertex1) - 1][int(vertex2) - 1] = weight
        direct_adj_matrix[int(vertex2) - 1][int(vertex1) - 1] = weight

    undirct_adj_matrix = [[1 if weight != 0 else 0 for weight in row] for row in direct_adj_matrix]

    direct_adj_matrix = np.matrix(direct_adj_matrix)
    undirct_adj_matrix = np.matrix(undirct_adj_matrix, dtype=int)

    return direct_adj_matrix, undirct_adj_matrix
