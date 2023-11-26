from core import LRE

import numpy as np

def get_test_matrix():
    adjacency_matrix = np.zeros((21, 21))
    job_list = [
        [1,2],
        [1,3],
        [1,4],
        [2,5],
        [2,7],
        [3,5],
        [3,6],
        [4,10],
        [5,7],
        [5,8],
        [5,20],
        [6,9],
        [6,10],
        [7,8],
        [7,11],
        [7,12],
        [8,20],
        [10,14],
        [10,20],
        [11,18],
        [12,15],
        [12,18],
        [13,15],
        [13,19],
        [14,19],
        [15,17],
        [15,18],
        [15,19],
        [15,21],
        [16,18],
        [16,21],
        [17,19],
        [17,21],
    ]
    for job in job_list:
        modify_matrix(adjacency_matrix, job[0]-1, job[1]-1)
    return np.asmatrix(adjacency_matrix)

def modify_matrix(matrix, i, j):
    matrix[i, j] = 1
    matrix[j, i] = 1

if __name__=="__main__":
    adjacency_matrix = get_test_matrix()

    re = LRE(adjacency_matrix)

    prob_matrix, re_matrix, r_matrix, s_matrix, similar_pairs = re.calculate()    