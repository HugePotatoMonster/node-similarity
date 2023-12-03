import dataset
from core import CN, Katz, LRE, RE
from model import SIModel

import numpy as np
import matplotlib.pyplot as plt

def calculate(adj_matrix, rate, days, label):

    cn = CN(adj_matrix)
    katz = Katz(adj_matrix)
    lre = LRE(adj_matrix)
    re = RE(adj_matrix, label)

    _, cn_pairs = cn.calculate()
    _, katz_pairs = katz.calculate(3)
    _, _, _, _, lre_pairs = lre.calculate()
    _, re_matrix, r_matrix, s_matrix, re_pairs = re.calculate(8)

    print(re_matrix)
    print(r_matrix)
    print(s_matrix)
    
    model = SIModel(adj_matrix, data_label=label)

    

    # model.calculate_spread(rate, days, True)
    # model.calculate_spread_avg(rate, days, repeat=10, save=True)

    # file_list = [
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-18-50.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-19-17.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-19-45.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-20-13.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-20-41.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-21-09.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-21-36.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-22-02.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-22-30.json",
    #     "model_f\\USAir\\0.125-40-2023-12-02_15-22-58.json"
    # ]

    file_list = [
        "model_f\\Jazz\\0.3-20-2023-12-02_18-33-40.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-33-44.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-33-49.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-33-53.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-33-57.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-34-01.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-34-05.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-34-09.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-34-13.json",
        "model_f\\Jazz\\0.3-20-2023-12-02_18-34-17.json"
    ]

    model.calculate_spread_avg_from_file(file_list)

    cn_dit = model.affected_ability(cn_pairs, alg_label="CN", save=False)
    katz_dit = model.affected_ability(katz_pairs, alg_label="Katz", save=False)
    lre_dit = model.affected_ability(lre_pairs, alg_label="LRE", save=False)
    re_dit = model.affected_ability(re_pairs, alg_label="RE", save=False)

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
    rate = 0.3
    days = 20

    nodes, edges, Jazz_adj_matrix = dataset.get_Jazz_network()
    USAir_direct_adj_matrix, USAir_undirct_adj_matrix = dataset.get_USAir_network()

    calculate(Jazz_adj_matrix, rate, days, "Jazz")