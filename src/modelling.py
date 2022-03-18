import sys
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

gamma_file = open('isend_data.txt')
gamma_list = gamma_file.readlines()
gamma_list = [str(el).strip() for el in gamma_list]
gamma_dic = {}
for g in gamma_list:
    pairs = g.split(" ")
    gamma_dic.setdefault(int(pairs[0]), float(pairs[1]))

SEGSIZE = 8192

scatter_algorithms = [
    # T = (P - 1) * (a + b*m)
    (1, "BASIC_LINEAR", "({3} - 1)*({0} + {1}*{2})"),

    # T = ceil(log2(P))*(ceil(log2(P) + 1))/2 *(a + mb) + log2(p)*a
    (2, "BINOMIAL",
     "(math.floor(math.log({3}, 2)) * {0} + {1}*{2}) + math.log({3}, 2)*{0}")
]

def linear(p, a, b, m):
    coff = (p - 1)
    return (coff * (a + m * b), coff)


def binomial(p, a, b, m):
    coff = math.floor(math.log(p, 2))
    return (coff * a + (p - 1) * m * b, coff)


def root_overhead(p, a, b):
    """root_overhead - is linear regression function betwen numer of processes and isend time.
    In Open MPI, each parent process in virtual topology sends message to its children using isend with wait_all
    procedure.
    """
    return a + p * b

def lin_reg(X, Y):
    """Huber regression is employed to buid linear regression.
    """
    if len(X) != len(Y):
        print("Linear regression: Length of arrays are different: {} - {}".format(len(X), len(Y)))
        return -1

    X = np.array(X)
    Y = np.array(Y)

    huber = HuberRegressor(fit_intercept=True, alpha=0.0,
                           max_iter=100, epsilon=1.35)

    huber.fit(X[:, np.newaxis], Y)

    return huber.intercept_, huber.coef_[0]

def exp_data_list(file_path, coll_type=0):
    """This method returns list of data from file.
    Experimental datas are numbers saved as rows.
    This method reads data from file and converts to a list.
    """
    data = []
    try:
        f = open(file_path, "r")
    except Exception as er:
        print(er)
    else:
        data = f.readlines()
        data = [row.strip().split()
                for row in data if len(row.strip().split()) > 0]
    if data:
        data = [[float(el) for el in row] for row in data]

    return data

def best_performance(data_list, coll_type):
   #The function returns list of best performance algorithms for  message size
    beg = 0
    # Number of collective algorithms for MPI_Bcast and MPI_Gather 
    #Scatter
    alg_count = 2 if coll_type else 6
    best_perf_alg = []
    while beg < len(data_list):
        best_alg = min(data_list[beg:beg + alg_count], key=lambda x: x[3])
        if best_alg:
            best_perf_alg.append(best_alg)
        beg += alg_count
    return best_perf_alg

def data_processing(data, a, b, _a, _b, message_sizes, times, alg_id=0, coll_type=0):
    """Processing data for linear regression
    """
    row = []
    ms = SEGSIZE
    for row in data:
        cond = (row[2] != 1)
        if alg_id:
            cond = (row[2] == alg_id)
        if cond:
            ns = row[1] / ms
            #if coll_type:
            coff = scatter_alg_cost(row[0], a, b, row[1], row[2])[1]
            #else:
               # coff = bcast_alg_cost(row[0], a, b, _a, _b, ns, ms, row[2])[
                    #1] + row[0] - 1

            if not coll_type:  # bcast
                if row[2] == -1:  # split-binary broadcast algorithm
                    message_sizes.append((coff * ms + row[1] / 2) / (coff + 1))
                    #message_sizes.append((coff * ms) / (coff + 1))
                    times.append(row[3] / (coff + 1))
                else:
                    message_sizes.append(ms)
                    times.append(row[3] / coff)
            else:  # scatter 
                if row[2] == 1: # or row[2] == 2:
                    message_sizes.append(row[1])
                    times.append(row[3] / coff)
                elif row[2] == 2:
                    message_sizes.append((row[1] * (row[0] - 1))/coff)
                    times.append(row[3] / coff)
                else:
                    message_sizes.append(row[1] / 2)
                    times.append(row[3] / (coff * 2))


def experimental_messages(data_list):
    '''This method extracts messages from hierarchical broadcast experiment data
    '''
    messages = []
    if not data_list:
        return messages
    mes = data_list[0][1]
    messages.append(mes)
    for el in data_list:
        if el[1] != mes:
            mes = el[1]
            messages.append(el[1])
    return messages

def scatter_alg_cost(p, a, b, m, alg_id):
    res = (0, 0)
    if alg_id == 1:
        res = scatter_linear(p, a, b, m)
    elif alg_id == 2:
        res = scatter_binomial(p, a, b, m)

    return res


def scatter_linear(p, a, b, m):
    coff = (p - 1)
    return (coff * (a + m * b), coff)

def scatter_binomial(p, a, b, m):
    coff = math.floor(math.log(p, 2))
    return (coff * a + (p - 1) * m * b, coff)


def ompi_optimal_scatter_alg(data_list):
    large_block_size = 92160
    intermediate_block_size = 6000
    small_block_size = 1024

    large_communicator_size = 60
    small_communicator_size = 10

    opt_scatter_algorithm = 1  # default value
    messages = experimental_messages(data_list)
    communicator_size = int(data_list[0][0])
    analy_estimation = []
    for message_size in messages:
        if message_size > large_block_size:
            opt_scatter_algorithm = 2
        elif message_size > intermediate_block_size:
            opt_scatter_algorithm = 2
        elif (communicator_size > large_communicator_size) or ((communicator_size > small_communicator_size) and (message_size < small_block_size)):
            opt_scatter_algorithm = 2

        analy_estimation.append((message_size, opt_scatter_algorithm))

    get_alg_exp = []
    for opalg in analy_estimation:
        for expdata in data_list:
            if expdata[2] == opalg[1] and expdata[1] == opalg[0]:
                get_alg_exp.append(expdata)
                break
    return get_alg_exp

def optimal_scatter_algorithm_by_model(hm_params, data_list):
    if len(data_list) == 0:
        print("Data list is empty!")
        return -1

    messages = experimental_messages(data_list)
    p = int(data_list[0][0])
    analy_estimation = []
    for m in messages:
        analytical_estimation = []
        for algorithmid in range(1, 2):
            value_of_combination = scatter_alg_cost(
                p, hm_params[algorithmid-1][0], hm_params[algorithmid-1][1], m, algorithmid)

            analytical_estimation.append(
                (algorithmid, value_of_combination, m))

        min_val = min(analytical_estimation, key=lambda x: x[1])
        analy_estimation.append(min_val)

    get_alg_exp = []
    for opalg in analy_estimation:
        for expdata in data_list:
            if expdata[2] == opalg[0] and expdata[1] == opalg[2]:
                get_alg_exp.append(expdata)
                break
    return get_alg_exp
