import os
import argparse
import modelling as exp
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rc('lines', linewidth=2)


def print_data_row(row):
    s = str(row[0]) + ' ' + str(row[1]) + " " + str(row[3]) + " "
    s += exp.scatter_algorithms[int(row[2] - 2)][1]
    return s.strip()


def selection_experiments(args):

    X = []
    Y = []

    isend_data = exp.exp_data_list('data/isend_data.txt')

    if not isend_data:
        print('Given file is not found!')
        return

    for it in isend_data:
        X.append(it[0])
        Y.append(it[1])

    lb_isend = exp.lin_reg(X, Y)
    vformat = ['\\num{' + "{:.1e}".format(v) + '}' for v in lb_isend]
    el = ", ".join(vformat)
    #    print("lb_isend: ");
    #    print(lb_isend)

    if args.ver == "4.1":
        train_data_path = 'data/sonic_long_' + str(args.nump)
    else:
        train_data_path = 'data/csi_long_' + str(args.nump)
    # converts data to list
    train_data_set = exp.exp_data_list(train_data_path)

    # Hockney model parameters are measured using collective algorithms
    hockney_model_parameters = []
    coll_algorithms = exp.scatter_algorithms
    if args.ver == "2.1":
        # Open MPI 2.1 does not contain Linear Non Blocking Algorithm
        coll_algorithms.pop()

    # Calculate latency and bandwidth using collective algorithms, iterate through algorithms
    for alg in coll_algorithms:
        X = []
        Y = []
        exp.data_processing(train_data_set, 0, 0,
                            lb_isend[0], lb_isend[1], X, Y, alg[0])
        hockney_model_parameters.append(exp.lin_reg(X, Y))
        print("Hockney model params: ")
        print(hockney_model_parameters[-1])
        value = hockney_model_parameters[-1]
        vformat = ['\\num{' + "{:.1e}".format(v) + '}' for v in value]
        el = ", ".join(vformat)

    if args.ver == "4.1":
        unseen_data_path = 'data/sonic_short_' + str(args.nump)
    else:
        unseen_data_path = 'data/csi_short_' + str(args.nump)
    if not unseen_data_path:
        print("Given file is not found!")
    unseen_data_set = exp.exp_data_list(unseen_data_path)

    unseen_data_set = [
        td for td in unseen_data_set if td[1] in range(65536, 1048576)]

    if not unseen_data_set:
        print('Unseen performance data does not exist!')
        return

    best_perf_alg = exp.best_performance(unseen_data_set, len(coll_algorithms))
    for el in best_perf_alg:
        print(print_data_row(el))

    print('----------------------------------------------------------------')

    model_opt_alg = exp.optimal_scatter_algorithm_by_model(
        hockney_model_parameters, unseen_data_set, len(coll_algorithms))

    for analy_est, best_alg in zip(model_opt_alg, best_perf_alg):
        print(print_data_row(analy_est), ' -- ',
              '{}%'.format(round(analy_est[3] / best_alg[3] * 100)))

    print('----------------------------------------------------------------')

    # Newer version of OMPI has a refined algorithm selection process

    if args.ver == "4.1":
        ompi_opt_alg = exp.new_ompi_optimal_scatter_alg(unseen_data_set)

    elif args.ver == "2.1":
        ompi_opt_alg = exp.ompi_optimal_scatter_alg(unseen_data_set)

    for ompi_alg, best_alg in zip(ompi_opt_alg, best_perf_alg):
        print(print_data_row(ompi_alg), ' -- ',
              '{}%'.format(round(ompi_alg[3] / best_alg[3] * 100)))

    Y_exp = []
    Y_model = []
    Y_ompi = []

    for alg1, alg2, alg3 in zip(best_perf_alg, model_opt_alg, ompi_opt_alg):
        Y_exp.append(alg1[3])
        Y_model.append(alg1[3])
        Y_ompi.append(alg3[3])
    data_types = ['Best',
                  'Model-based',
                  'Open MPI'
                  ]
    colors = {
        'Best': 'limegreen',
        'Model-based': 'orangered',
        'Open MPI': 'dodgerblue'
    }
    linestyle = {
        'Best': '-',
        'Model-based': '--',
        'Open MPI': '--'
    }

    lw = 4
    X = np.array(exp.experimental_messages(unseen_data_set))
    Y_exp = np.array(Y_exp)
    Y_model = np.array(Y_model)
    Y_ompi = np.array(Y_ompi)
    Y = []
    Y.append(Y_exp)
    Y.append(Y_model)
    Y.append(Y_ompi)
    i = 0

    title_font = {'family': 'monospace',
                  'color': 'black',
                  'weight': 'normal',
                  }

    for name in data_types:
        plt.plot(
            X,
            Y[i],
            linestyle=linestyle[name],
            linewidth=lw,
            color=colors[name],
            label=data_types[i])
        legend = plt.legend(
            loc='upper left',
            frameon=False,
            title='P = ' + str(args.nump),
            prop=dict(size='small'))
        i += 1

    plt.xscale('log', base=2)
    plt.xlabel('Message sizes (Bytes)')  # , fontsize=23)
    plt.ylabel('Time (seconds)')  # , fontsize=23)
    coll_name = 'MPI_Scatter'
    plt.title(f'{coll_name} P = {args.nump}', fontdict=title_font)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nump", required=True, help="Number of Processes")
    parser.add_argument("--ver", required=True, help="Open MPI Version, Supported: 2.1, 4.1")
    args = parser.parse_args()
    args.path = os.getcwd()
    selection_experiments(args)
