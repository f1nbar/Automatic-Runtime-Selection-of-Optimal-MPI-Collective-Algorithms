# Automatic Runtime Selection of Optimal MPI Collective Algorithms - Final Year Project
## Finbar Ó Deaghaidh

This repository contains all programs used for my final year project, described in detail within
the report. Some programs are based on work by [Emin Nuriyev.](https://csgitlab.ucd.ie/emin.nuri/mpicollmodelling/-/tree/master/)

The `src` directory contains the following programs: 


1. `scatter_perf_2.1.c`
    This C program contains the communication experiments used to measure the performance of the 
    Open MPI collective algorithms on the CSI Cluster. The targeted version of Open MPI is version
    2.1. 

2. `scatter_perf_4.1.c` This program contains the communications experiments targeted for UCD Sonic HPC and Open MPI version 4.1.        
    
    These programs can be compiled with the following command ` mpicc -o scatter scatter_X.1.c -lm -lgsl -lgslcblas -Wall` and replacing X with the targeted version of Open MPI. Mpirun is needed to run the compiled programs along with GSL (GNU Scientific Library).


3. `modelling.py` is a Python program that contains all key logic for defining the analytical performance models, estimiating model parameters and determining the best, model chosen and Open MPI chosen algorithm. 

4. `algorithm_selection.py` handles all data from the communication experiments and passes the relevant information to `modelling.py`. This program also gives the user detailed information on the chosen algorithms for the best case, model estimated and OMPI. The estimated model parameters are also displayed.

This program can be executed as follows: `python algorithm_selection.py --nump 14 --ver 2.1`


<details><summary>Example Results</summary>

```
python algorithm_selection.py --nump 20 --ver 2.1
Hockney model parameters for BASIC_LINEAR algorithm
(1.4207391754478532e-05, 1.804368002504216e-09)
Hockney model parameters for BINOMIAL algorithm
(0.001579767352605663, 1.2833073739083567e-09)
----------------------------------------------------------------
Best Perf
20.0 65536.0 0.0082967827 BINOMIAL
20.0 122880.0 0.0085273748 BINOMIAL
20.0 245760.0 0.0121899695 BASIC_LINEAR
20.0 483328.0 0.0186999238 BINOMIAL
20.0 860160.0 0.0303659596 BASIC_LINEAR
20.0 1048576.0 0.0350860307 BINOMIAL
----------------------------------------------------------------
Model Perf
20.0 65536.0 0.0102159298 BINOMIAL  --  100%
20.0 122880.0 0.009906621 BASIC_LINEAR  --  116%
20.0 245760.0 0.0121899695 BASIC_LINEAR  --  100%
20.0 483328.0 0.0205014815 BASIC_LINEAR  --  110%
20.0 860160.0 0.0314007564 BINOMIAL  --  103%
20.0 1048576.0 0.0350860307 BINOMIAL  --  100%
----------------------------------------------------------------
OMPI Perf
20.0 65536.0 0.0102159298 BASIC_LINEAR  --  123%
20.0 122880.0 0.009906621 BASIC_LINEAR  --  116%
20.0 245760.0 0.0121899695 BASIC_LINEAR  --  100%
20.0 483328.0 0.0205014815 BASIC_LINEAR  --  110%
20.0 860160.0 0.0303659596 BASIC_LINEAR  --  100%
20.0 1048576.0 0.0357208696 BASIC_LINEAR  --  102%
```

</details>
