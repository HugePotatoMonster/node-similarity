import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def disease_spread_model_c(adj_matrix, beta, days, start_node):
    num_nodes = len(adj_matrix)

    infected = np.zeros(num_nodes, dtype=int)
    infected[start_node] = 1
    
    # Initialize a list to store infected nodes over time
    infected_nodes = [set({start_node})]
    
    for _ in range(days):
        temp = np.zeros(num_nodes, dtype=int)
        for node in range(num_nodes):
            if infected[node] == 1:
                neighbors = np.nonzero(adj_matrix[node, :])[1]
                for neighbor in neighbors:
                    if infected[neighbor] == 0:
                        if np.random.rand() < beta:
                            temp[neighbor] = 1
        
        infected = infected|temp
        infected_nodes.append(set(np.where(infected > 0)[0]))
    
    return infected_nodes

def plot_infection_process(adj_matrix, infected_nodes, days):
    G = nx.from_numpy_matrix(adj_matrix)
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

    animation = FuncAnimation(fig, update, frames=min(days, len(infected_nodes)), repeat=False, interval=500)

    plt.show()
