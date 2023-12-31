import dataset
from core import CN, Katz, LRW, LRE, RE
from model import SIModel

import numpy as np
import matplotlib.pyplot as plt


def calculate(adj_matrix, rate, days, label):
    # cn = CN(adj_matrix)
    katz = Katz(adj_matrix)
    lrw = LRW(adj_matrix)
    lre = LRE(adj_matrix)
    re = RE(adj_matrix, label)

    # _, cn_pairs = cn.calculate()
    _, katz_pairs = katz.calculate(3)
    _, lrw_pairs = lrw.calculate(3)
    _, _, _, _, lre_pairs = lre.calculate()
    _, _, _, _, re_pairs = re.calculate(8)

    # print("CN mutual: {}".format(calculate_mutual_ratio(cn_pairs)))
    print("Katz mutual: {}".format(calculate_mutual_ratio(katz_pairs)))
    print("LRW mutual: {}".format(calculate_mutual_ratio(lrw_pairs)))
    print("LRE mutual: {}".format(calculate_mutual_ratio(lre_pairs)))
    print("RE mutual: {}".format(calculate_mutual_ratio(re_pairs)))

    model = SIModel(adj_matrix, data_label=label)

    # model.calculate_spread(rate, days, True)
    model.calculate_spread_avg(rate, days, repeat=15, save=True)

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

    # file_list = [
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-33-40.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-33-44.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-33-49.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-33-53.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-33-57.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-34-01.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-34-05.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-34-09.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-34-13.json",
    #     "model_f\\Jazz\\0.3-20-2023-12-02_18-34-17.json",
    # ]

    # model.calculate_spread_avg_from_file(file_list)

    # cn_dit = model.affected_ability(cn_pairs, alg_label="CN", save=False)
    katz_dit = model.affected_ability(katz_pairs, alg_label="Katz", save=False)
    lrw_dit = model.affected_ability(lrw_pairs, alg_label="LRW", save=False)
    lre_dit = model.affected_ability(lre_pairs, alg_label="LRE", save=False)
    re_dit = model.affected_ability(re_pairs, alg_label="RE", save=False)

    # katz_dit = model.affected_ability_avg(katz_pairs, rate, days, 10, alg_label="Katz", save=True)
    # lre_dit = model.affected_ability_avg(lre_pairs, rate, days, 10, alg_label="LRE", save=True)
    # re_dit = model.affected_ability_avg(re_pairs, rate, days, 10, alg_label="RE", save=True)

    # plt.plot(cn_dit, label="CN")
    plt.plot(katz_dit, label="Katz")
    plt.plot(lrw_dit, label="LRW")
    plt.plot(lre_dit, label="LRE")
    plt.plot(re_dit, label="RE")
    # plt.plot(re_dit, label="LRE")
    # plt.plot(lre_dit, label="RE")

    plt.title(label)
    plt.xlabel("Days")
    plt.ylabel("DIT(t)")

    plt.legend()

    plt.show()


def calculate_mutual_ratio(pairs):
    mutual_points = 0

    for i in range(len(pairs)):
        j = pairs[i]
        if j < len(pairs) and pairs[j] == i:
            mutual_points += 1

    mutual_ratio = mutual_points / len(pairs)
    return mutual_ratio


if __name__ == "__main__":
    rate = 0.3
    days = 20

    Karate_adj_matrix = dataset.get_Karate_network()
    nodes, edges, Jazz_adj_matrix = dataset.get_Jazz_network()
    USAir_direct_adj_matrix, USAir_undirct_adj_matrix = dataset.get_USAir_network()
    Email_adj_matrix = dataset.get_Email_network()

    # calculate(Karate_adj_matrix, rate, days, "Karate")
    calculate(Jazz_adj_matrix, rate, days, "Jazz")
    # calculate(USAir_undirct_adj_matrix, rate, days, "USAir")
    # calculate(Email_adj_matrix, rate, days, "Email")
