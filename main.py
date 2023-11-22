import dataset
import core

nodes, edges, adjacency_matrix = dataset.get_Jazz_network()
prob_matrix = core.prob(adjacency_matrix)
core.re_matrix(prob_matrix)