import numpy as np

from core import LRE, RE, CN, LRW
import dataset


def modify_matrix(matrix, i, j):
    matrix[i, j] = 1
    matrix[j, i] = 1


def get_test_matrix():
    adjacency_matrix = np.zeros((21, 21))
    job_list = [
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [2, 7],
        [3, 5],
        [3, 6],
        [4, 10],
        [5, 7],
        [5, 8],
        [5, 20],
        [6, 9],
        [6, 10],
        [7, 8],
        [7, 11],
        [7, 12],
        [8, 20],
        [10, 14],
        [10, 20],
        [11, 18],
        [12, 15],
        [12, 18],
        [13, 15],
        [13, 19],
        [14, 19],
        [15, 17],
        [15, 18],
        [15, 19],
        [15, 21],
        [16, 18],
        [16, 21],
        [17, 19],
        [17, 21],
    ]
    for job in job_list:
        modify_matrix(adjacency_matrix, job[0] - 1, job[1] - 1)
    return np.asmatrix(adjacency_matrix)


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
    # adj_matrix = np.matrix(
    #     [
    #         [0, 1, 1, 0, 0, 0],
    #         [1, 0, 1, 0, 0, 0],
    #         [1, 1, 0, 1, 0, 0],
    #         [0, 0, 1, 0, 1, 0],
    #         [0, 0, 0, 1, 0, 1],
    #         [0, 0, 0, 0, 1, 0],
    #     ]
    # )
    adj_matrix = dataset.get_Email_network()
    # cn = CN(adj_matrix)
    # cn.calculate()
    # adj_matrix = get_test_matrix()
    # re = RE(adj_matrix)
    # _, _, _, _, re_pairs = re.calculate(5)
    # print(re_pairs)
