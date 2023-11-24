import numpy as np

def prob(adj_matrix):
    num_nodes = len(adj_matrix)
    max_degree = int(np.max(np.sum(adj_matrix, axis=1)))

    prob_matrix = np.zeros((num_nodes, max_degree + 1), dtype=np.float64)

    for i in range(num_nodes):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        
        degrees_i = [np.sum(adj_matrix[j]) for j in [i] + list(neighbors)]    
        degrees_i.sort(reverse=True)
        degrees_i.extend([0] * (max_degree + 1 - len(degrees_i)))
        degrees_i /= np.sum(degrees_i)
        
        prob_matrix[i, :len(degrees_i)] = degrees_i

    return prob_matrix

def relative_entropy(prob_1, prob_2):
    ans=0
    for i in range(prob_1.shape[0]):
        if prob_1[i]==0 or prob_1[i]==0:
            return ans
        ans += prob_1*np.log(prob_1[i]/prob_2[i])

    return ans

def re_matrix(prob_matrix):
    num_nodes = prob_matrix.shape[0]
    re_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(num_nodes):
            re_matrix[i][j] = relative_entropy(prob_matrix[i], prob_matrix[j])
    return re_matrix