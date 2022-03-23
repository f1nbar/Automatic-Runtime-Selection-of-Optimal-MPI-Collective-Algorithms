import os
import argparse
import modelling as exp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 23})

def print_data_row(row, coll):
    s = str(row[0]) + ' ' + str(row[1]) + " " + str(row[3]) + " "
    s += exp.scatter_algorithms[int(row[2] - 2)][1]
    return s.strip()

def selection_experiments(args):

    print(args)

    X = []
    Y = []

    ised_data = exp.exp_data_list(
        'data/isend_data.txt', 1)

    if not ised_data:
        print('Given file is not found!')
        return

    for it in ised_data:
        X.append(it[0])
        Y.append(it[1])

    lb_isend = exp.lin_reg(X, Y)
    vformat = ['\\num{' + "{:.1e}".format(v) + '}' for v in lb_isend]
    el = ", ".join(vformat)
    print("lb_isend: ");
    print(lb_isend)
    train_data_path = 'data/6_local_long.txt'# + str(args.nump)
    coll_type = 1
    #converts data to list
    train_data_set = exp.exp_data_list(train_data_path, coll_type)

    # Hockney model parameters are measured using collective algorithms
    hockney_model_parameters = []
    coll_algorithms = []
    if not coll_type:
        coll_algorithms = exp.scatter_algorithms
    else:
        coll_algorithms = exp.scatter_algorithms
        # change for future algorithm

    # Calculate latency and bandwidth using collective algorithms, iterate through algorithms
    for alg in coll_algorithms:
        X = []
        Y = []
        exp.data_processing(train_data_set, 0, 0,
                            lb_isend[0], lb_isend[1], X, Y, alg[0], coll_type)
        hockney_model_parameters.append(exp.lin_reg(X, Y))
        print("Hockney model params: ")
        print(hockney_model_parameters[-1])
        value = hockney_model_parameters[-1]
        vformat = ['\\num{' + "{:.1e}".format(v) + '}' for v in value]
        el = ", ".join(vformat)

    if coll_type:
        unseen_data_path = 'data/6_local_short.txt'
    else:
        unseen_data_path = 'data/short.txt'

    unseen_data_set = exp.exp_data_list(unseen_data_path, coll_type)


    unseen_data_set = [
        td for td in unseen_data_set if td[1] in range(65536, 827382)]

    if not unseen_data_set:
        print('Unseen performance data does not exist!')
        return

    best_perf_alg = exp.best_performance(unseen_data_set, coll_type)
    for el in best_perf_alg:
        print(print_data_row(el, 1))

    print('----------------------------------------------------------------')

  #  if coll_type:
    model_opt_alg = exp.optimal_scatter_algorithm_by_model(
            hockney_model_parameters, unseen_data_set)
   # else:
   #     model_opt_alg = exp.optimal_bcast_algorithm_by_model(
   #          hockney_model_parameters, lb_isend[0], lb_isend[1], unseen_data_set)

    for analy_est, best_alg in zip(model_opt_alg, best_perf_alg):
        print(print_data_row(analy_est, 1), ' -- ',
              '{}%'.format(round(analy_est[3]/best_alg[3] * 100)))
        #print(print_data_row(analy_est, coll_type))

    print('----------------------------------------------------------------')
    #if coll_type:
    ompi_opt_alg = exp.ompi_optimal_scatter_alg(unseen_data_set)
    #else:
    #    ompi_opt_alg = exp.ompi_optimal_bcast_alg(unseen_data_set)
    for ompi_alg, best_alg in zip(ompi_opt_alg, best_perf_alg):
        print(print_data_row(ompi_alg, 1), ' -- ',
              '{}%'.format(round(ompi_alg[3]/best_alg[3] * 100)))
        #print(print_data_row(ompi_alg, coll_type))
    Y_exp = []
    Y_model = []
    Y_ompi = []

    for alg1, alg2, alg3 in zip(best_perf_alg, model_opt_alg, ompi_opt_alg):
        Y_exp.append(alg1[3])
        Y_model.append(alg2[3])
        Y_ompi.append(alg3[3])
    data_types = ['Best',
                  'Model-based',
                  'Open MPI'
                  ]
    colors = {
        'Best': 'green',
        'Model-based': 'r',
        'Open MPI': 'blue'
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
    #They are not the same size..,46 and 15
    #print(len(X))
    #print(len(Y[2])k)
    i = 0

    title_font = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 20,
                  }

    for name in data_types:
        plt.plot(
            X,
            Y[i],
            color=colors[name],
            linestyle=linestyle[name],
            linewidth=lw,
            label=data_types[i])
        legend = plt.legend(
            loc='upper left',
            frameon=False,
            title= 'P = ' + str(args.nump),
            prop=dict(size='x-small'))
        i += 1
    plt.xscale('log', base=2)
    plt.xlabel('Message sizes (Bytes)')  # , fontsize=23)
    plt.ylabel('Time(sec)')  # , fontsize=23)
    coll_name = 'MPI_Scatter' if coll_type else 'MPI_Bcast'
    plt.title(f'{coll_name} P = {args.nump}', fontdict=title_font)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nump", required=True, help="Number of processes")
    args = parser.parse_args()
    args.path = os.getcwd()
    selection_experiments(args)


