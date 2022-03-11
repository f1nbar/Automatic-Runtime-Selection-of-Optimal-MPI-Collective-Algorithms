"""Hierarchical optimization is topology oblivious optimization methodology.
"""
import sys
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
"""Broadcast algorithms structure in Open MPI gives us algorithms ID. We can set algorithms for given communicator
using those IDs.
"""

gamma_file = open('data/isend/grisou.txt')
gamma_list = gamma_file.readlines()
gamma_list = [str(el).strip() for el in gamma_list]
gamma_dic = {}
for g in gamma_list:
    pairs = g.split(" ")
    gamma_dic.setdefault(int(pairs[0]), float(pairs[1]))


SEGSIZE = 8192
bcast_algorithms = [
    (1, "BASIC_LINEAR", "{5}*({4} - 1)*({0} + {1}*{2}*{3})"),
    (2, "K-CHAIN", "(math.ceil(({4} - 1)/4) + {5}*{2} - 1)*({0} + {1}*{3})"),
    (3, "PIPELINE", "{5}*({4} + {2} - 2)*({0} + {1}*{3})"),
    (4, "SPLIT_BINARY_TREE",
     "{5}*(math.floor(math.log({4},2)) + {2}/2 - 1)*({0} + {1}*{3}) + {0} + ({1}*{2}/2)*{3}"),
    (5, "BINARY_TREE",
     "{5}*(math.floor(math.log({4},2)) + {2} - 1)*({0} + {1}*{3})"),
    (6, "BINOMIAL",
     "{5}*(math.floor(math.log({4},2)) + {2} - 1)*({0} + {1}*{3})")
]

gather_algorithms = [
    # T = (P - 1) * (a + b*m)
    (1, "BASIC_LINEAR", "({3} - 1)*({0} + {1}*{2})"),

    # T = ceil(log2(P))*(ceil(log2(P) + 1))/2 *(a + mb) + log2(p)*a
    (2, "BINOMIAL",
     "(math.floor(math.log({3}, 2)) * {0} + {1}*{2}) + math.log({3}, 2)*{0}"),

    # T = a + (P - 1) * (2*a + b*m)
    (3, "LINEAR_SYNC", "{0} + ({3} - 1)*(2*{0} + {1}*{2})")
]


#### Gather  algorithms ####
def garther_linear(p, a, b, m):
    coff = (p - 1)
    return (coff * (a + m * b), coff)


def gather_binomial(p, a, b, m):
    coff = math.floor(math.log(p, 2))
    #return (coff * (2*a + m * b), coff)
    return (coff * a + (p - 1) * m * b, coff)


def gather_linear_sync(p, a, b, m):
    coff = (p - 1)
    return (coff * (2 * a + m * b), coff)


def gather_alg_cost(p, a, b, m, alg_id):
    res = (0, 0)
    if alg_id == 1:
        res = garther_linear(p, a, b, m)
    elif alg_id == 2:
        res = gather_binomial(p, a, b, m)
    elif alg_id == 3:
        res = gather_linear_sync(p, a, b, m)

    return res


def root_overhead(p, a, b):
    """root_overhead - is linear regression function betwen numer of processes and isend time.
    In Open MPI, each parent process in virtual topology sends message to its children using isend with wait_all
    procedure.
    """
    #return gamma_dic[p]
    return a + p * b


def alg_root_overhead(alg_id, p, a, b):
    """This method returns root overhead for given algorithm"""
    res = 1
    if alg_id == 2:  # K-Chain
        res = root_overhead(5, a, b) / root_overhead(2, a, b)
    elif alg_id == 4:  # Split-binary tree
        res = root_overhead(3, a, b) / root_overhead(2, a, b)
    elif alg_id == 5:  # Binary tree
        res = root_overhead(3, a, b) / root_overhead(2, a, b)
    elif alg_id == 6:  # Binomial tree
        root_overhead(math.floor(math.log(p, 2)), a, b) / \
            root_overhead(2, a, b)

    return res


def basic_linear(p, a, b, ns, ms):
    coff = p - 1
    return (coff * (a + ms * ns * b), coff)


def k_chain_tree(p, a, b, _a, _b, ns, ms):
    roverhead = alg_root_overhead(2, p, _a, _b)
    coff = roverhead * ns + math.ceil((p - 1) / 4) - 1
    return (coff * (a + ms * b), coff)


def pipelined_tree(p, a, b, ns, ms):
    coff = p + ns - 2
    return (coff * (a + ms * b), coff)


def split_binary_tree(p, a, b, _a, _b, ns, ms):
    roverhead = alg_root_overhead(4, p, _a, _b)
    coff = roverhead * (math.floor(math.log(p + 1, 2)) + ns / 2 - 1)
    return (coff * (a + ms * b) + a + (ms * ns * b) / 2, coff)


def binary_tree(p, a, b, _a, _b, ns, ms):
    roverhead = 1
    if ns > 1:
        roverhead = alg_root_overhead(5, p, _a, _b)
    coff = roverhead * (math.floor(math.log(p + 1, 2)) + ns - 1)
    return (coff * (a + ms * b), coff)


def binomial_tree(p, a, b, _a, _b, ns, ms):
    root_child = math.ceil(math.log(p, 2))
    h = math.floor(math.log(p, 2))
    pmax = int(min(h, ns))
    s = 0
    roverhead = alg_root_overhead(6, p, _a, _b)
    for i in range(1, pmax - 1):
        isend_overhead_i = root_overhead(
            root_child - i, _a, _b) / root_overhead(2, _a, _b)
        s += isend_overhead_i
    s += (pmax - 1) * roverhead
    s += (abs(h - ns) + 1) * roverhead
    res = (s * (a + ms * b), s)
    coff = (math.floor(math.log(p, 2)) * ns)
    #if ns == 1:
    #    coff = math.floor(math.log(p, 2))
    res = (coff * (a + ms * b), coff)
    return res


def bcast_alg_cost(p, a, b, _a, _b, ns, ms, alg_id):
    res = (0, 0)
    if alg_id == 1:
        res = basic_linear(p, a, b, ns, ms)
    elif alg_id == 2:
        res = k_chain_tree(p, a, b, _a, _b, ns, ms)
    elif alg_id == 3:
        res = pipelined_tree(p, a, b, ns, ms)
    elif alg_id == 4:
        res = split_binary_tree(p, a, b, _a, _b, ns, ms)
    elif alg_id == 5:
        res = binary_tree(p, a, b, _a, _b, ns, ms)
    elif alg_id == 6:
        res = binomial_tree(p, a, b, _a, _b, ns, ms)
    return res


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
            # segmentation is used in broadcast algorithms
            ns = row[1] / ms
            if coll_type:
                coff = gather_alg_cost(row[0], a, b, row[1], row[2])[1]
            else:
                coff = bcast_alg_cost(row[0], a, b, _a, _b, ns, ms, row[2])[
                    1] + row[0] - 1

            if not coll_type:  # bcast
                if row[2] == -1:  # split-binary broadcast algorithm
                    message_sizes.append((coff * ms + row[1] / 2) / (coff + 1))
                    #message_sizes.append((coff * ms) / (coff + 1))
                    times.append(row[3] / (coff + 1))
                else:
                    message_sizes.append(ms)
                    times.append(row[3] / coff)
            else:  # gather
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


def optimal_bcast_algorithm_by_model(hm_params, _a, _b, data_list):
    if len(data_list) == 0:
        print("Data list is empty!")
        return -1

    messages = experimental_messages(data_list)
    p = int(data_list[0][0])
    opt_algs = []

    ms = SEGSIZE
    for m in messages:
        ns = m / ms
        analytical_estimation = []
        for algorithmid in range(1, 7):
            if algorithmid == 4:
                algorithmid = 5
            value_of_combination = bcast_alg_cost(
                p, hm_params[algorithmid-1][0], hm_params[algorithmid-1][1], _a, _b, ns, ms, algorithmid)[0]

            value_of_combination += (p - 1) * \
                (hm_params[1][0] + SEGSIZE * hm_params[1][1])
            analytical_estimation.append((algorithmid, value_of_combination, m))

        min_val = min(analytical_estimation, key=lambda x: x[1])
        opt_algs.append(min_val)

    opt_hi_bcast = []
    for opt_alg in opt_algs:
        for exp_alg in data_list:
            if exp_alg[1] == opt_alg[2] and exp_alg[2] == opt_alg[0]:
                opt_hi_bcast.append(exp_alg)
                break

    return opt_hi_bcast



def optimal_hibcast_algorithm_by_model(hm_params, _a, _b, data_list):
    if len(data_list) == 0:
        print("Data list is empty!")
        return -1

    messages = experimental_messages(data_list)
    p = int(data_list[0][0])
    opt_algs = []

    ms = SEGSIZE
    for m in messages:
        ns = m / ms
        analytical_estimation = []
        for algorithmid in range(1, 7):
            if algorithmid == 4:
                algorithmid = 5
            value_of_combination = bcast_alg_cost(
                p, hm_params[algorithmid-1][0], hm_params[algorithmid-1][1], _a, _b, ns, ms, algorithmid)[0]

            value_of_combination += (p - 1) * \
                (hm_params[1][0] + SEGSIZE * hm_params[1][1])
            analytical_estimation.append((algorithmid, value_of_combination, m))

        min_val = min(analytical_estimation, key=lambda x: x[1])
        opt_algs.append(min_val)

    opt_hi_bcast = []
    for opt_alg in opt_algs:
        for exp_alg in data_list:
            if exp_alg[1] == opt_alg[2] and exp_alg[2] == opt_alg[0]:
                opt_hi_bcast.append(exp_alg)
                break

    return opt_hi_bcast


def best_performance(data_list, coll_type):
    """The function returns list of best performance algorithms for  message size
    """
    beg = 0
    # Number of collective algorithms for MPI_Bcast and MPI_Gather 
    alg_count = 3 if coll_type else 6
    best_perf_alg = []
    while beg < len(data_list):
        best_alg = min(data_list[beg:beg + alg_count], key=lambda x: x[3])
        if best_alg:
            best_perf_alg.append(best_alg)
        beg += alg_count
    return best_perf_alg


'''def execute_mpi_code(mpi_cmd, p_start, p_end, p_stride):
    """This method is used to call MPI code to execute.
    """
    # hbcast_cmd = "mpirun -map-by ppr:1:socket -mca pml ob1 -mca btl tcp,self -mca coll_tuned_use_dynamic_rules 1 -hostfile ~/ngrisou48 -n {} broadcast_gather"
    for p in range(p_start, p_end + 1, p_stride):
        subproc = sp.Popen(mpi_cmd.format(p), shell=True)
        while subproc.poll() is None:
            # Process hasn't exited yet, let's wait some
            time.sleep(0.5)
        output, error = subproc.communicate()
        if subproc.returncode != 0:
            print("MPI program failed {} {} {}".format(
                subproc.returncode, output, error))
            sys.exit()
'''

def exp_data_list(file_path, coll_type=0):
    """This method returns list of data from file.
    Experimental datas are numbers saved as rows.
    This method reads data from file and convsert it to the list.
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


def ompi_optimal_bcast_alg(data_list):
    """This code originaly implemented in Open MPI to make a decision which broadcast algorithm and segment size
    should be selected. We are using only to compute segment size for different message size and communicator size.
    """
    small_message_size = 2048
    intermediate_message_size = 370728
    a_p16 = 3.2118e-6
    b_p16 = 8.7936
    a_p64 = 2.3679e-6
    b_p64 = 1.1787
    a_p128 = 1.6134e-6
    b_p128 = 2.1102
    opt_bcast_algorithm = 3
    messages = experimental_messages(data_list)
    communicator_size = int(data_list[0][0])
    opt_alg = []
    for message_size in messages:
        if message_size < small_message_size:
            opt_bcast_algorithm = 6
        elif message_size < intermediate_message_size:
            opt_bcast_algorithm = 4
        elif communicator_size < a_p128 * message_size + b_p128:
            opt_bcast_algorithm = 3
        elif communicator_size < 13:
            opt_bcast_algorithm = 4
        elif communicator_size < a_p64 * message_size + b_p64:
            opt_bcast_algorithm = 3
        elif communicator_size < a_p16 * message_size + b_p16:
            opt_bcast_algorithm = 3
        else:
            opt_bcast_algorithm = 3

        opt_alg.append((message_size, opt_bcast_algorithm))

    ompi_hi_bcast = []
    for oalg in opt_alg:
        for hbcast in data_list:
            if hbcast[1] == oalg[0] and hbcast[2] == oalg[1]:
                ompi_hi_bcast.append(hbcast)
                break
    return ompi_hi_bcast


def optimal_gather_algorithm_by_model(hm_params, data_list):
    if len(data_list) == 0:
        print("Data list is empty!")
        return -1

    messages = experimental_messages(data_list)
    p = int(data_list[0][0])
    analy_estimation = []
    for m in messages:
        analytical_estimation = []
        for algorithmid in range(1, 4):
            value_of_combination = gather_alg_cost(
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


def ompi_optimal_gather_alg(data_list):
    large_block_size = 92160
    intermediate_block_size = 6000
    small_block_size = 1024

    large_communicator_size = 60
    small_communicator_size = 10

    opt_gahter_algorithm = 1  # default value
    messages = experimental_messages(data_list)
    communicator_size = int(data_list[0][0])
    analy_estimation = []
    for message_size in messages:
        if message_size > large_block_size:
            opt_gahter_algorithm = 3
        elif message_size > intermediate_block_size:
            opt_gahter_algorithm = 3
        elif (communicator_size > large_communicator_size) or ((communicator_size > small_communicator_size) and (message_size < small_block_size)):
            opt_gahter_algorithm = 2

        analy_estimation.append((message_size, opt_gahter_algorithm))

    get_alg_exp = []
    for opalg in analy_estimation:
        for expdata in data_list:
            if expdata[2] == opalg[1] and expdata[1] == opalg[0]:
                get_alg_exp.append(expdata)
                break
    return get_alg_exp


def bcast_alg_graph(data_set, alg_id, xtype):
    X = []
    Y = []

    for row in data_set:
        if row[2] == alg_id:
            if xtype:
                X.append(row[1]/SEGSIZE)
            else:
                X.append(row[0])
            Y.append(row[3])
            if row[0] in range(26, 70):
                print(X[-1], Y[-1])

    huber = lin_reg(X, Y)

    if xtype:
        print('Linear function of ns, a = {}, b = {}'.format(
            huber[0], huber[1]))

    plt.plot(
        X,
        Y,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Alg_id = {}'.format(alg_id))
    #legend_title = 'Performance of algorithms'
    legend = plt.legend(
        loc='upper left',
        frameon=False,
        # title=legend_title,
        prop=dict(size='x-small'))

    #plt.xscale('log', basex=2)
    if xtype:
        plt.xlabel('Number of segment sizes')
    else:
        plt.xlabel('Number of processors')
    plt.ylabel('Time(sec)')
    plt.grid(True)
    plt.show()

