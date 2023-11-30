import path

import numpy as np
import heapq
from tqdm import trange
from scipy.stats import linregress
import os

class CN:
    def __init__(self, adj_matrix):
        undirct_adj_matrix = [[1 if weight != 0 else 0 for weight in row] for row in adj_matrix.A]
        self.adj_matrix = np.matrix(undirct_adj_matrix, dtype=int)

    def calculate(self):
        cn_matrix = np.linalg.matrix_power(self.adj_matrix, 2).A

        similar_pairs = np.argmax(cn_matrix, axis=1)
        return cn_matrix, similar_pairs

    
class Katz:
    def __init__(self, adj_matrix):
        undirct_adj_matrix = [[1 if weight != 0 else 0 for weight in row] for row in adj_matrix.A]
        self.adj_matrix = np.matrix(undirct_adj_matrix, dtype=int)
        max_eigenvalue = max(np.linalg.eigvals(self.adj_matrix).real)
        self.beta = 1/(2*max_eigenvalue)

    def calculate(self, order):
        katz_matrix = np.zeros(self.adj_matrix.shape)
        for i in range(order):
            katz_matrix += self.beta**i*np.linalg.matrix_power(self.adj_matrix, i)
        np.fill_diagonal(katz_matrix, 0)

        similar_pairs = np.argmax(katz_matrix, axis=1)
        return katz_matrix, similar_pairs
    
class LRE:
    def __init__(self, adj_matrix):
        undirct_adj_matrix = [[1 if weight != 0 else 0 for weight in row] for row in adj_matrix.A]
        self.adj_matrix = np.matrix(undirct_adj_matrix, dtype=int)

    def calculate(self):
        prob_matrix = self.__prob(self.adj_matrix)
        lre_matrix = self.__lre_matrix(prob_matrix)
        r_matrix = self.__r_matrix(lre_matrix)
        s_matrix = self.__s_matrix(r_matrix)

        similar_pairs = np.argmax(s_matrix, axis=1)

        return prob_matrix, lre_matrix, r_matrix, s_matrix, similar_pairs
    
    def __prob(self, adj_matrix):
        num_nodes = len(adj_matrix)
        max_degree = int(np.max(np.sum(adj_matrix, axis=1)))

        prob_matrix = np.zeros((num_nodes, max_degree + 1), dtype=np.float64)

        for i in range(num_nodes):
            neighbors = np.nonzero(adj_matrix.A[i] == 1)[0]
            
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

    def __lre_matrix(self, prob_matrix):
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
                elif r_max==0:
                    s_matrix[i][j] = 1
                else:
                    s_matrix[i][j] = 1 - r_matrix[i][j]/r_max
        return s_matrix
class RE:
    def __init__(self, adj_matrix, filename="") -> None:
        self.adj_matrix = adj_matrix

        if filename != "":
            root_path = os.path.join(os.getcwd(), "ns")
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            self.json_path = os.path.join(os.getcwd(), "ns", filename+".json")
        else:
            self.json_path = ""
        self.shortest_paths_matrix = None
        self.graph_radius = 0
        self.fractal_dimension = 0

    def calculate(self, lb):
        self.__all_pairs_shortest_paths()
        self.__fractal_dimension(lb)
        prob_matrix = self.__prob(self.adj_matrix)
        re_matrix = self.__re_matrix(prob_matrix)
        r_matrix = self.__r_matrix(re_matrix)
        s_matrix = self.__s_matrix(r_matrix)

        similar_pairs = np.argmax(s_matrix, axis=1)

        return prob_matrix, re_matrix, r_matrix, s_matrix, similar_pairs

    def __dijkstra(self, start):
        num_nodes = self.adj_matrix.shape[0]
        dist = {node: float('infinity') for node in range(num_nodes)}
        dist[start] = 0

        priority_queue = [(0, start)]

        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)

            if current_dist > dist[current_node]:
                continue

            for neighbor, weight in enumerate(self.adj_matrix.A[current_node]):
                if weight > 0:
                    new_dist = dist[current_node] + weight
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heapq.heappush(priority_queue, (new_dist, neighbor))

        return dist

    def __all_pairs_shortest_paths(self):
        num_nodes = self.adj_matrix.A.shape[0]
        self.shortest_paths_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            dist_from_i = self.__dijkstra(i)
            self.shortest_paths_matrix[i, :] = [dist_from_i[j] for j in range(num_nodes)]

        max_distances = np.max(self.shortest_paths_matrix, axis=1)
        self.graph_radius = np.min(max_distances)

        return self.shortest_paths_matrix
    
    def __rearrange_nodes(self, graph):
        strengths = np.sum(graph, axis=1)

        nodes_with_strengths = list(enumerate(strengths))

        nodes_with_strengths.sort(key=lambda x: x[1], reverse=True)

        sorted_nodes = [node for node, _ in nodes_with_strengths]

        return sorted_nodes
    
    def __box_num(self, lb):
        box_num = 0

        dual_network = np.where(self.shortest_paths_matrix >= lb, self.shortest_paths_matrix, 0)

        num_nodes = self.adj_matrix.shape[0]
        colors = [-1] * num_nodes
        sorted_nodes = self.__rearrange_nodes(dual_network)

        for i in range(num_nodes):
            node = sorted_nodes[i]
            if i==0:
                colors[node] = box_num
                box_num += 1
            else:
                neighbor_colors = set(colors[n] for n in range(num_nodes) if dual_network[node][n] != 0)

                for j in range(i):
                    color = colors[sorted_nodes[j]]
                    if color not in neighbor_colors and color!=-1:
                        colors[node] = color
                        break
                else:
                    colors[node] = box_num
                    box_num += 1
        return box_num
    
    def __fractal_dimension(self, lb):
        ns = []

        if self.json_path != "":
            s_ns_map = path.load_ns(self.json_path)
        else:
            s_ns_map = {}
        
        for s in trange(2,lb):
            if str(s) not in s_ns_map:
                box_num = self.__box_num(s)
                ns.append(box_num)
                s_ns_map[s] = box_num
            else:
                ns.append(s_ns_map[str(s)])

        if self.json_path != "":
            path.save_ns(s_ns_map, self.json_path)

        ln_ns = np.log(ns)
        ln_s = np.log([s for s in range(2,lb)])

        slope, _, _, _, _ = linregress(ln_s, ln_ns)
        self.fractal_dimension = -slope
    
    def __prob(self, adj_matrix):
        num_nodes = len(adj_matrix)

        local_dims = np.zeros(num_nodes)

        for i in range(num_nodes):
            num_points_at_distance_r = np.count_nonzero(self.shortest_paths_matrix[i] == self.graph_radius)
            num_points_within_distance_r = np.count_nonzero(self.shortest_paths_matrix[i] <= self.graph_radius)-1
            local_dims[i] = self.graph_radius * (num_points_at_distance_r / num_points_within_distance_r) if num_points_within_distance_r != 0 else 0

        max_degree = int(np.max(np.sum(adj_matrix, axis=1)))
        prob_matrix = np.zeros((num_nodes, max_degree + 1), dtype=np.float64)

        for i in range(num_nodes):
            neighbors = np.nonzero(adj_matrix.A[i] == 1)[0]
            
            dim_i = [np.sum(local_dims[j]) for j in [i] + list(neighbors)]    
            dim_i.sort(reverse=True)
            dim_i.extend([0] * (max_degree + 1 - len(dim_i)))
            dim_i /= np.sum(dim_i)
            
            prob_matrix[i, :len(dim_i)] = dim_i

        return prob_matrix

    def __tsalli_entropy(self, prob_1, prob_2):
        ans=0
        for i in range(prob_1.shape[0]):
            if prob_1[i]==0 or prob_2[i]==0:
                return ans
            r = prob_1[i]/prob_2[i]
            ans += (r**self.graph_radius-r)/-self.graph_radius
        return ans

    def __re_matrix(self, prob_matrix):
        num_nodes = prob_matrix.shape[0]
        re_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for i in range(num_nodes):
            for j in range(num_nodes):
                re_matrix[i][j] = self.__tsalli_entropy(prob_matrix[i], prob_matrix[j])
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
                elif r_max==0:
                    s_matrix[i][j] = 1
                else:
                    s_matrix[i][j] = 1 - r_matrix[i][j]/r_max
        return s_matrix