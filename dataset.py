import re
import networkx as nx
import numpy as np


def get_Karate_network():
    number_pairs = []
    with open("datasets\\Karate.wl", "r") as file:
        for line in file:
            pairs = re.findall(r"\[(\d+)\s(\d+)\]", line)
            if pairs:
                number_pairs.extend([(int(pair[0]), int(pair[1])) for pair in pairs])

    all_values = [value for pair in number_pairs for value in pair]
    num_nodes = max(all_values)

    adj_matrix = np.zeros((num_nodes, num_nodes))

    for edge in number_pairs:
        adj_matrix[edge[0] - 1, edge[1] - 1] = 1
        adj_matrix[edge[1] - 1, edge[0] - 1] = 1

    return np.matrix(adj_matrix, dtype=int)


def get_Jazz_network():
    with open("datasets\\Jazz-Musicians-Network.wl", "r") as file:
        data = file.read()

    nodes_match = re.search(r"\{(.+?)\}", data)
    edges_match = re.findall(r"UndirectedEdge\[(\d+),\s*(\d+)\]", data)

    nodes_str = nodes_match.group(1)

    nodes = list(map(int, re.findall(r"\d+", nodes_str)))

    edges = [tuple(map(int, edge)) for edge in edges_match]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    adj_matrix = nx.to_numpy_matrix(G, nodelist=nodes)

    return nodes, edges, adj_matrix


def get_USAir_network():
    edges_content = []
    with open("datasets\\US-Air-Lines.wl", "r") as file:
        in_edges_section = False
        for line in file:
            if line.startswith("*Edges"):
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

    undirct_adj_matrix = [
        [1 if weight != 0 else 0 for weight in row] for row in direct_adj_matrix
    ]

    direct_adj_matrix = np.matrix(direct_adj_matrix)
    undirct_adj_matrix = np.matrix(undirct_adj_matrix, dtype=int)

    return direct_adj_matrix, undirct_adj_matrix


def get_Email_network():
    edges = []
    with open("datasets\\email-Eu-core.wl", "r") as file:
        for line in file:
            edge = tuple(map(int, line.strip().split()))
            edges.append(edge)

    num_nodes = max(max(edge) for edge in edges) + 1
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    for edge in edges:
        node1, node2 = edge
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1

    return np.matrix(adj_matrix)
