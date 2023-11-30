import dataset
from core import CN, Katz, LRE, RE
from model import SIModel

import numpy as np
import matplotlib.pyplot as plt

def calculate(adj_matrix, label):

    cn = CN(adj_matrix)
    katz = Katz(adj_matrix)
    lre = LRE(adj_matrix)
    re = RE(adj_matrix, label)

    _, cn_pairs = cn.calculate()
    _, katz_pairs = katz.calculate(3)
    _, _, _, _, lre_pairs = lre.calculate()
    _, _, _, _, re_pairs = re.calculate(8)
    
    model = SIModel(adj_matrix, data_label=label)

    rate = 0.125
    days = 40

    model.calculate_spread(rate, days)

    cn_dit = model.affected_ability(cn_pairs, alg_label="CN", save=True)
    katz_dit = model.affected_ability(katz_pairs, alg_label="Katz", save=True)
    lre_dit = model.affected_ability(lre_pairs, alg_label="LRE", save=True)
    re_dit = model.affected_ability(re_pairs, alg_label="RE", save=True)

    # katz_dit = model.affected_ability_avg(katz_pairs, rate, days, 10, alg_label="Katz", save=True)
    # lre_dit = model.affected_ability_avg(lre_pairs, rate, days, 10, alg_label="LRE", save=True)
    # re_dit = model.affected_ability_avg(re_pairs, rate, days, 10, alg_label="RE", save=True)

    plt.plot(cn_dit, label='CN')
    plt.plot(katz_dit, label='Katz')
    plt.plot(lre_dit, label='LRE')
    plt.plot(re_dit, label='RE')

    plt.title(label)
    plt.xlabel('Days')
    plt.ylabel('DIT(t)')

    plt.legend()

    plt.show()

if __name__=="__main__":
    nodes, edges, Jazz_adj_matrix = dataset.get_Jazz_network()
    USAir_direct_adj_matrix, USAir_undirct_adj_matrix = dataset.get_USAir_network()

    calculate(USAir_undirct_adj_matrix, "USAir")