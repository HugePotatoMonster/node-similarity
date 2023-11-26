import numpy as np

class LRE:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix

    def calculate(self):
        prob_matrix = self.__prob(self.adj_matrix)
        re_matrix = self.__re_matrix(prob_matrix)
        r_matrix = self.__r_matrix(re_matrix)
        s_matrix = self.__s_matrix(r_matrix)

        similar_pairs = np.argmax(s_matrix, axis=1)

        return prob_matrix, re_matrix, r_matrix, s_matrix, similar_pairs
    
    def __prob(self, adj_matrix):
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

    def __relative_entropy(self, prob_1, prob_2):
        ans=0
        for i in range(prob_1.shape[0]):
            if prob_1[i]==0 or prob_2[i]==0:
                return ans
            ans += prob_1[i]*np.log(prob_1[i]/prob_2[i])

        return ans

    def __re_matrix(self, prob_matrix):
        num_nodes = prob_matrix.shape[0]
        re_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for i in range(num_nodes):
            for j in range(num_nodes):
                re_matrix[i][j] = self.__relative_entropy(prob_matrix[i], prob_matrix[j])
        return re_matrix

    def __r_matrix(self, re_matrix):
        num_nodes = re_matrix.shape[0]
        r_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for i in range(num_nodes):
            for j in range(num_nodes):
                r_matrix[i][j] = re_matrix[i][j]+re_matrix[j][i]
        return r_matrix

    def __s_matrix(self, r_matrix):
        num_nodes = r_matrix.shape[0]
        s_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        r_max = np.max(r_matrix)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i==j:
                    s_matrix[i][j] = 0
                    continue
                s_matrix[i][j] = 1 - r_matrix[i][j]/r_max
        return s_matrix
    
class Katz:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        max_eigenvalue = max(np.linalg.eigvals(self.adj_matrix))
        self.beta = 1/(2*max_eigenvalue)

    def calculate(self, order):
        katz_matrix = np.zeros(self.adj_matrix.shape)
        for i in range(order):
            katz_matrix += self.beta**i*np.linalg.matrix_power(self.adj_matrix, i)
        np.fill_diagonal(katz_matrix, 0)

        similar_pairs = np.argmax(katz_matrix, axis=1)
        return katz_matrix, similar_pairs