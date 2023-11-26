import dataset
from core import Katz, LRE
from model import ISModel
import test

import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    nodes, edges, adjacency_matrix = dataset.get_Jazz_network()

    katz = Katz(adjacency_matrix)
    lre = LRE(adjacency_matrix)

    katz_matrix, katz_pairs = katz.calculate(3)
    prob_matrix, re_matrix, r_matrix, s_matrix, lre_pairs = lre.calculate()
    
    model = ISModel(adjacency_matrix, data_label="Jazz")

    rate = 0.1
    days = 40

    model.calculate_spread(rate, days)

    katz_dit = model.affected_ability(katz_pairs, alg_label="Katz", save=True)
    lre_dit = model.affected_ability(lre_pairs, alg_label="LRE", save=True)

    plt.plot(katz_dit, label='katz_dit')
    plt.plot(lre_dit, label='lre_dit')

    plt.title('DIT')
    plt.xlabel('Days')
    plt.ylabel('DIT(t)')

    plt.legend()

    plt.show()