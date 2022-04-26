import os
import argparse
import modelling as exp
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rc('axes', labelsize=18)

def print_data_row(row):
    s = str(row[0]) + ' ' + str(row[1]) + " " + str(row[3]) + " "
    s += exp.scatter_algorithms[int(row[2] - 2)][1]
    return s.strip()


def selection_experiments(args):

    if args.ver == "4.1":
        train_data_path = 'data/sonic_long_' + str(args.nump)
    else:
        train_data_path = 'data/csi_long_' + str(args.nump)
    # converts data to list
    train_data_set = exp.exp_data_list(train_data_path)

    coll_algorithms = exp.scatter_algorithms
    if args.ver == "2.1":
        # Open MPI 2.1 does not contain Linear Non Blocking Algorithm
        coll_algorithms.pop()

    # Calculate hockney model parameters for each algorithm
    hockney_model_parameters = []
    for alg in coll_algorithms:
        X = []
        Y = []
        exp.data_processing(train_data_set, 0, 0,
                            X, Y, alg[0])
        hockney_model_parameters.append(exp.lin_reg(X, Y))
        print("Hockney model parameters for" ,alg[1], "algorithm")
        print(hockney_model_parameters[-1])

    if args.ver == "4.1":
        unseen_data_path = 'data/sonic_short_' + str(args.nump)
    else:
        unseen_data_path = 'data/csi_short_' + str(args.nump)
    if not unseen_data_path:
        print("Given file is not found!")
    unseen_data_set = exp.exp_data_list(unseen_data_path)

    unseen_data_set = [
        td for td in unseen_data_set if td[1] in range(65536, 1048576)] #Data range where rendezvous protocol is used

    if not unseen_data_set:
        print('Unseen performance data does not exist!')
        return

    print('----------------------------------------------------------------')
    print('Best Perf')

    best_perf_alg = exp.best_performance(unseen_data_set, len(coll_algorithms))
    for el in best_perf_alg:
        print(print_data_row(el))

    print('----------------------------------------------------------------')
    print('Model Perf')

    model_opt_alg = exp.optimal_scatter_algorithm_by_model(
        hockney_model_parameters, unseen_data_set, len(coll_algorithms))

    for analy_est, best_alg in zip(model_opt_alg, best_perf_alg):
        print(print_data_row(analy_est), ' -- ',
              '{}%'.format(round(analy_est[3] / best_alg[3] * 100)))

    print('----------------------------------------------------------------')
    print('OMPI Perf')

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
        Y_model.append(alg2[3])
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

    X = np.array(exp.extract_messages(unseen_data_set))
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
            linewidth=6,
            color=colors[name],
            label=data_types[i])
        i += 1

    plt.xscale('log', base=2)
    plt.xlabel('Message sizes (Bytes)', fontsize=20)
    plt.ylabel('Time (seconds)', fontsize=20)
    coll_name = 'MPI_Scatter'
    plt.title(f'{coll_name} P = {args.nump}', fontdict=title_font, fontsize=20)
    plt.legend(
        loc='upper left',
        frameon=False,
        prop=dict(size='20'))
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('axes', labelsize=20)
    plt.savefig('/home/finbar/compsci/fyp/git/final-year-project/images/myfig.png',bbox_inches="tight", pad_inches=0.1, dpi=1000)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nump", required=True, help="Number of Processes")
    parser.add_argument("--ver", required=True, help="Open MPI Version, Supported: 2.1, 4.1")
    args = parser.parse_args()
    args.path = os.getcwd()
    selection_experiments(args)
