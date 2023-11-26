import path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import trange

class ISModel:
    def __init__(self, adj_matrix, data_label="D"):
        self.adj_matrix = adj_matrix
        self.__node_num = adj_matrix.shape[0]
        self.__data_label = data_label
        self.__f = None
        self.__rate = 1
        self.__days = 0

    def __spread(self, beta, days, start_node):
        num_nodes = len(self.adj_matrix)

        infected = np.zeros(num_nodes, dtype=int)
        infected[start_node] = 1
        
        infected_nodes = [set({start_node})]
        
        for _ in range(days):
            temp = np.zeros(num_nodes, dtype=int)
            for node in range(num_nodes):
                if infected[node] == 1:
                    neighbors = np.nonzero(self.adj_matrix[node, :])[1]
                    for neighbor in neighbors:
                        if infected[neighbor] == 0 and np.random.rand() < beta:
                            temp[neighbor] = 1
            
            infected = infected|temp
            infected_nodes.append(set(np.where(infected > 0)[0]))
        
        return infected, infected_nodes
    
    def calculate_spread(self, rate, days):
        self.__rate = rate
        self.__days = days

        # spread
        self.__f = np.zeros((self.__node_num, days+1))
        for i in trange(self.__node_num):
            self.__f[i, :] = [len(s) for s in self.__spread(self.__rate, self.__days, i)[1]]
    
    def affected_ability(self, pairs, alg_label="A", save=False):
        dif = []

        for day in range(self.__days+1):
            d_list = []
            for i in range(self.__node_num):
                d_list.append(np.abs((self.__f[i][day]-self.__f[pairs[i]][day])/self.__node_num))
            dif.append(np.var(d_list))

        # save
        if save:
            path.save_result(dif, f"{self.__data_label}-{alg_label}")

        return dif
            

    def plot_infection_process(self, infected_nodes, days):
        G = nx.from_numpy_matrix(self.adj_matrix)
        pos = nx.spring_layout(G)

        fig, ax = plt.subplots(figsize=(12, 8))

        def update(day):
            ax.clear()

            infected_set = infected_nodes[day]
            non_infected_nodes = list(G.nodes - infected_set)

            nx.draw_networkx_nodes(G, pos, nodelist=non_infected_nodes, node_color='blue', node_size=1)
            nx.draw_networkx_nodes(G, pos, nodelist=list(infected_set), node_color='red', node_size=1)
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

            ax.set_title(f'Infection Process - Day {day}')

        FuncAnimation(fig, update, frames=min(days, len(infected_nodes)), repeat=False, interval=500)

        plt.show()
