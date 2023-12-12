import numpy as np

from core import LRE, RE, CN, LRW
import dataset

if __name__ == "__main__":
    # adj_matrix = np.matrix(
    #     [
    #         [0, 0.86, 0.4, 0, 0, 0],
    #         [0.86, 0, 0.5, 0, 0, 0],
    #         [0.4, 0.5, 0, 0.1, 0, 0],
    #         [0, 0, 0.1, 0, 0.05, 0],
    #         [0, 0, 0, 0.05, 0, 0.3],
    #         [0, 0, 0, 0, 0.3, 0],
    #     ]
    # )
    adj_matrix = np.matrix(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    # adj_matrix = dataset.get_Karate_network()
    # cn = CN(adj_matrix)
    # cn.calculate()
    re = LRW(adj_matrix)
    lrw_matrix, similar_pairs = re.calculate(5)
    print(lrw_matrix)
    print(similar_pairs)
