import math
import numpy as np
from sklearn.linear_model import HuberRegressor

scatter_algorithms = [
    # T = (P - 1) * (a + b*m)
    (1, "BASIC_LINEAR", "({3} - 1)*({0} + {1}*{2})"),
    # T = ceil(log2(P))*(ceil(log2(P) + 1))/2 *(a + mb) + log2(p)*a
    (2, "BINOMIAL", "(math.floor(math.log({3}, 2)) * {0} + {1}*{2}) + math.log({3}, 2)*{0}"),
    # T = (P - 1) * (a + b*m)
    # Linear Non Blocking algorithm added in newer versions of Open MPI
    (3, "LINEAR_NB", "({3} - 1)*({0} + {1}*{2})"),

]

def data_processing(data, a, b, message_sizes, times, alg_id=0):
    # Processing data for linear regression
    row = []
    for row in data:
        cond = (row[2] != 1) #ALG ID = 1
        if alg_id:
            cond = (row[2] == alg_id)
        if cond:
            #print("alg_cost: ", alg_cost, "row[0]: ", row[0], " ,a ,b ",a," ",b," ",row[1], " ", row[2])
            if row[2] == 1: #Linear 
                alg_cost = scatter_alg_cost(row[0], a, b, row[1], row[2])[1]
                message_sizes.append(row[1]) #append message size from data list
                times.append(row[3] / alg_cost) #divide time elapsed by cost
            elif row[2] == 2: #Binomial
                alg_cost = scatter_alg_cost(row[0], a, b, row[1], row[2])[1]
                message_sizes.append((row[1] * (row[0] - 1))/ alg_cost) #Message x log2P x P-1 for inorder binomial tree
                times.append(row[3] / alg_cost)
            else: #Linear NB
                alg_cost = scatter_alg_cost(row[0], a, b, row[1], row[2])[1]
                message_sizes.append(row[1])
                times.append(row[3] / alg_cost)


def lin_reg(X, Y):
    #Huber regression to calculate linear regression for alpha and beta
    if len(X) != len(Y):
        print("Length of arrays must be the same: {} - {}".format(len(X), len(Y)))
        return -1
    X = np.array(X)
    Y = np.array(Y)
    huber = HuberRegressor(fit_intercept=True, alpha= 0.0,
                           max_iter=100, epsilon=1.40)
    huber.fit(X[:, np.newaxis], Y)

    return huber.intercept_, huber.coef_[0]


def exp_data_list(file_path):
    #Convert row data from file to list
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

def extract_messages(data_list):
    #Extracts messages from commiication experiments
    messages = []
    if not data_list:
        return messages
    mess = data_list[0][1]
    messages.append(mess)
    for el in data_list:
        if el[1] != mess:
            mess = el[1]
            messages.append(el[1])
    return messages


def best_performance(data_list, alg_count):
    # The function returns list of best performance algorithms for  message size, obtained from results
    beg = 0
    best_perf_alg = []
    while beg < len(data_list):
        best_alg = min(data_list[beg:beg + alg_count], key=lambda x: x[3])
        if best_alg:
            best_perf_alg.append(best_alg)
        beg += alg_count
    return best_perf_alg

#In Open MPI, each parent process in virtual topology sends message to its children using isend with wait_all procedure.
def calc_root_overhead(p, a, b):
    return a + p * b


def scatter_alg_cost(p, a, b, m, alg_id):
    res = (0, 0)
    if alg_id == 1:
        res = scatter_linear(p, a, b, m)
    elif alg_id == 2:
        res = scatter_binomial(p, a, b, m)
    elif alg_id == 3:
        res = scatter_linear_nb(p, a, b, m)

    return res

#Cost of performance models
def scatter_linear(p, a, b, m):
    coff = (p - 1)
    return coff * (a + m * b), coff

def scatter_binomial(p, a, b, m):
    coff = math.floor(math.log(p, 2))
    return coff * a + (p - 1) * m * b, coff

def scatter_linear_nb(p, a, b, m):
    coff = (p - 1)
    return coff * (a + m * b), coff


def new_ompi_optimal_scatter_alg(data_list):
    # Reimplementation of the Open MPI fixed decision algorithm, version 4.1
    messages = extract_messages(data_list)
    communicator_size = int(data_list[0][0])
    analy_estimation = []

    for message_size in messages:
        if communicator_size < 4:
            if message_size < 2:
                opt_scatter_algorithm = 3
            elif message_size < 131072:
                opt_scatter_algorithm = 1
            elif message_size < 262144:
                opt_scatter_algorithm = 3
            else:
                opt_scatter_algorithm = 1
        elif communicator_size < 8:
            if message_size < 2048:
                opt_scatter_algorithm = 2
            elif message_size < 4096:
                opt_scatter_algorithm = 1
            elif message_size < 8192:
                opt_scatter_algorithm = 2
            elif message_size < 32768:
                opt_scatter_algorithm = 1
            elif message_size < 1048576:
                opt_scatter_algorithm = 3
            else:
                opt_scatter_algorithm = 1
        elif communicator_size < 16:
            if message_size < 16384:
                opt_scatter_algorithm = 2
            elif message_size < 1048576:
                opt_scatter_algorithm = 3
            else:
                opt_scatter_algorithm = 1
        elif communicator_size < 32:
            if message_size < 16384:
                opt_scatter_algorithm = 2
            elif message_size < 32768:
                opt_scatter_algorithm = 1
            else:
                opt_scatter_algorithm = 3
        elif communicator_size < 64:
            if message_size < 512:
                opt_scatter_algorithm = 2
            elif message_size < 8192:
                opt_scatter_algorithm = 3
            elif message_size < 16384:
                opt_scatter_algorithm = 2
            else:
                opt_scatter_algorithm = 3
        else:
            if message_size < 512:
                opt_scatter_algorithm = 2
            else:
                opt_scatter_algorithm = 3

        analy_estimation.append((message_size, opt_scatter_algorithm))
    get_alg_exp = []
    for opalg in analy_estimation:
        for expdata in data_list:
            if expdata[2] == opalg[1] and expdata[1] == opalg[0]:
                get_alg_exp.append(expdata)
                break
    return get_alg_exp


def ompi_optimal_scatter_alg(data_list):
    # Reimplementation of the Open MPI fixed decision algorithm, version 2.1
    messages = extract_messages(data_list)
    communicator_size = int(data_list[0][0])
    analy_estimation = []
    small_block_size = 300
    small_comm_size = 10

    for message_size in messages:

        if (communicator_size > small_comm_size) and (message_size < small_block_size):
            opt_scatter_algorithm = 2 #Binomial Algorithm
        else:
            opt_scatter_algorithm = 1 #Basic Linear

        analy_estimation.append((message_size, opt_scatter_algorithm))

    get_alg_exp = []
    for opalg in analy_estimation:
        for expdata in data_list:
            if expdata[2] == opalg[1] and expdata[1] == opalg[0]:
                get_alg_exp.append(expdata)
                break
    return get_alg_exp


def optimal_scatter_algorithm_by_model(hockney_params, data_list, alg_count):
    #Predicts the optimal scatter algorithm using the analytical models
    if len(data_list) == 0:
        print("Data list is empty!")
        return -1
    messages = extract_messages(data_list)
    p = int(data_list[0][0])
    analy_estimation = []
    for m in messages:
        analytical_estimation = []
        for algorithmid in range(1, alg_count+1):
            value_of_combination = scatter_alg_cost(
                p, hockney_params[algorithmid - 1][0], hockney_params[algorithmid - 1][1], m, algorithmid)
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
