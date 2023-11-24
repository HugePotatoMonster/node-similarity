import dataset
import core

if __name__=="__main__":
    nodes, edges, adjacency_matrix = dataset.get_Jazz_network()
    prob_matrix = core.prob(adjacency_matrix)
    re_reamtrix = core.re_matrix(prob_matrix)