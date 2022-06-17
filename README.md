# Bayesian Network - SAT-based Local Improvement Method


[![DOI](https://zenodo.org/badge/336274070.svg)](https://zenodo.org/badge/latestdoi/336274070)


_These instructions have been tested on linux
 (last updated: 17th June 2022)_

> Originally accepted at AAAI-21. The tags [cwidth] and [expert] refer to the 
> two follow-up versions accepted at NeurIPS-21 and UAI-22 resp.

## Required programming languages and software

* `Java SDK` (required to run BLIP)
    - `Maven` (required to build BLIP)
* `Python 3.6` or higher (required to run BN-SLIM)
* `C++` (required to run Merlin) [cwidth]
* `Make` (required to build Merlin and UWrMaxSat)
* `git` (required to obtain ETL source code)
* `Python 2.7` (required to run ETL)

> **Note:** Before you start, move all python (.py) files into the `slim` directory:
  (See Section: Directory Structure)


## External tools

### UWrMaxSat (`setup-uwrmaxsat.sh`)

1. Download and unzip source from [MaxSAT Evaluation 2019][1]

    ```sh
    pushd solvers
    wget https://maxsat-evaluations.github.io/2019/mse19-solver-src/complete/UWrMaxSat-1.0.zip
    unzip UWrMaxSat-1.0.zip
    ```


2. Apply patch `uwrmaxsat.patch` to fix minor bug

    ```sh
    pushd UWrMaxSat-1.0/code
    patch -ub uwrmaxsat/MsSolver.cc -i ../../uwrmaxsat.patch
    ```

3. Build UWrMaxSat
    
    ```sh
    chmod 777 starexec_build
    ./starexec_build
    popd
    ```

4. Place built binary in `solvers` directory

    ```sh
    cp UWrMaxSat-1.0/bin/uwrmaxsat . -v
    popd
    ```

5. (Optional) Test

    ```sh
    solvers/uwrmaxsat -m test.cnf -v1 -cpu-lim=2
    ```


### BLIP (`setup-blip.sh`)

1. Download and unzip source from [BLIP][2]

    ```sh
    pushd solvers
    wget https://ipg.idsia.ch/upfiles/blip-publish.zip --no-check-certificate
    unzip blip-publish.zip
    ```

2. Apply patch `blip.patch`

    ```sh
    pushd blip-publish
    patch -ub -p1 -i ../blip.patch
    ```

3. Compile BLIP [cwidth] and move jar to `solvers/blip.jar`

    ```sh
    ./compile-blip.sh
    cp blip.jar ../blip.jar -v
    ```

5. Undo patch `blip.patch` and apply `blip-con.patch`

    ```sh
    patch -u -p1 -i ../blip.patch --reverse
    patch -ub -p1 -i ../blip-con.patch
    ```

6. Compile BLIP [expert] and move jar to `solvers/blip-con.jar`

    ```sh
    ./compile-blip.sh
    cp blip.jar ../blip-con.jar -v
    popd
    popd
    ```

7. (Optional) Tests

    a. [cwidth] check if `java -jar solvers/blip.jar solver.kg.adv` contains `-cw`
    b. [expert] check if `java -jar solvers/blip.jar solver.kg.adv` contains `-con`
    c. Check if `tmp.res` contains `elim-order` field after running

    ```sh
    java -jar solvers/blip.jar solver.kmax -j test.jkl -r tmp.res -w 5 -v 1
    ```


### Merlin (`setup-merlin.sh`) [cwidth]

1. Download and unzip source from [Merlin][8]

    ```sh
    pushd solvers
    wget https://github.com/radum2275/merlin/tree/008d910307a2043f9446676f28010079ea49ec15 --no-check-certificate -O merlin-src.zip
    unzip merlin-src.zip
    ```

2. Apply patch `merlin.patch` to enable elim-order output

    ```sh
    pushd merlin-src
    patch -ub -p1 -i ../merlin.patch
    ```

3. Compile Merlin

    ```sh
    make
    popd
    ```

4. Place executable in `solvers` directory

    ```sh
    cp merlin-src/bin/merlin . -v
    popd
    ```

5. (Optional) Test and check if `tmp.PR.json` contains `value` of `-7.730569` after running

    ```sh
    solvers/merlin -t PR -a wmb -M lex -o tmp -O json -i 4 -f test.uai -e test.evid
    ```


### ET-Learn (`download-etlearn.sh`)

1. Download and unzip source from [et-learn][3] or clone the [git repo][4]

    ```sh
    pushd solvers
    git clone https://github.com/marcobb8/et-learn.git et_learn
    ```
2. Follow instructions for installing `et-learn` from [readme][5]

> **Note:** `et-learn` requires Python 2.7 while BN-SLIM requires Python 3.6+ and 
> these are not mutually compatible. Hence it is recommended to use [virtual
> environments][6] (also refer to [this][7]) and keep the two softwares separate.

2. Copy driver file `hcet.py` to same directory

    ```sh
    mv ../hcet.py et_learn
    ```

3. (Optional) Test and check if `tmp.res` is generated after running

    ```sh
    cd solvers/et_learn
    python hcet.py ../../test.data 5 -p -o tmp
    ```


## Datasets and Experiment Data

The datasets are available in the `datasets` folder, which contains subfolders
`dat`, `jkl` and `con` (see Section on File formats for more details).

The Experimental results are contained in the `experiments` folder.
The raw data from the experiments are provided as `.csv` files
while the scatter plots are available in the `plots` subfolder.


## Requirements for BN-SLIM

To install requirements:

```setup
pip install -r requirements.txt
```

To install optional requirements that allow drawing graphs etc.:

```setup
apt install graphviz
pip install -r optional_requirements.txt
```


## Directory Structure

The directory structure should now be as follows (`.zip` files hidden for brevity, 
`*` denotes the newly generated)

```
bnslim
├── README.md
├── setup-uwrmaxsat.sh
├── setup-blip.sh
├── setup-merlin.sh
├── download-etlearn.sh
├── test.cnf
├── test.jkl
├── test.dat
├── test.uai
├── test.evid
├── test.data
├── demo.res
├── demo.net
├── requirements.txt
├── optional_requirements.txt
├── solvers
│   ├── uwrmaxsat.patch
│   ├── blip.patch
│   ├── merlin.patch
│   ├── UWrMaxSat-1.0*
│   ├── uwrmaxsat*
│   ├── blip-publish*
|   ├── blip.jar*
│   ├── merlin-src*
│   ├── merlin*
|   └── et_learn*
|       ├── hcet.py
│       └── ... 
├── datasets*
|   ├── dat/
|   └── jkl/
├── experiments*
|   ├── baseline_data.csv
|   ├── heuristic_data.csv
│   └── plots/
└── slim
    ├── blip.py
    ├── bnio.py
    ├── complexity_encoding.py
    ├── berg_encoding.py
    ├── eval_model.py
    ├── samer_veith.py
    ├── slim.py
    ├── utils.py
    └── verify_cwidth.py
```


## Running BN-SLIM

```sh
cd slim
# bounded treewidth
python slim.py ../test.jkl 5 -v -u kmax -t 60
# bounded msss
python slim.py ../test.jkl 0 -v -u kmax -t 60 -b 0 -d ../test.dat --start-with ../demo.res -w --feasible-cw --feasible-cw-threshold 108
```

> Run `python slim.py --help` for description of available options

> For larger networks it is recommended to invoke as `python -O slim.py ...`


## File formats

Since most of the file formats used/required for this software and its
auxiliary helper softwares to run aren't standard, we provide a brief 
description of some of these file formats.

### `.res` file type

A `.res` file contains the DAG expressed by means of the parent sets
as well as a field indicating the score of the DAG and optionally
the elimination ordering used to bound the treewidth of the DAG. 
> See `demo.res` for an example.

The patch applied to BLIP allows it to output `.res` files with the 
elimination ordering which isn't supported by it out-of-the-box.
Additionally, it adds support for working with maximum state space
size instead of treewidth.

The patch applied to Merlin allows us to customize the elimination ordering
technique/heuristic of the variables via the `-M` command line argument.
In our use-case, we pre-set the ordering in the uai file and supply `-M lex`
to invoke the lexicographic ordering, thereby preserving and respecting
our pre-supplied ordering.


### `.dat`/`.data` file types

These files contain the samples drawn from the Bayesian Networks. This 
isn't a standard format and hence there exist discrepancies in the way
different programs and applications treat them. In general, these 
are space-separated (or comma-separated) files with or without the header.
The header refers to two lines at the beginning of the file.
The first line of the header contains the names of the random
variables (space-separated). The second line contains the domain sizes
of the variables in the same order.

### `.jkl` file type

A `.jkl` file contains the pre-computed parent set score function caches.
Usually these are computed via the `blip.jar scorer.is ...`.

### `.uai`/`.net` file types

These are somewhat standard file formats for storing Bayesian Networks.
The `.uai` format is required by Merlin and the `.net` format is used
as an intermediate format to go from `.res -> .uai`.

### `.evid` file type

File format for supplying evidence to Merlin.

The provided python script `hcet.py` allows one to obtain the solution of the
ETL algorithm(s) in the form of a compliant `.res` file which can be supplied
to `slim.py` through the `--start-with` option. 
Run `python et_learn/hcet.py --help` for more details.

> **Note:** Since the DAG computed by ETL could contain parent sets that 
> are absent from the pre-computed score function caches in datasets/jkl,
> one might need to "merge" the jkl files output when running `hcet.py` with
> the one from dataset/jkl to obtain a new jkl file which can then be used as
> input for BN-SLIM


## Results

### NeurIPS 2021 paper

`experiments/baseline_data.csv` and `experiments/heuristic_data.csv`
contain all the data for the 16 datasets from `datasets.zip`, 
for multiple treewidths, msss bounds and random seeds
run with the following configuration:

```sh
python -O slim.py <dataset> <treewidth> -t5400 -u <heuristic> -d <datfile.dat> --start-with <heur_sol.res> -w --feasible-cw --feasible-cw-threshold <msssbound> -r<seed> --budget 0 -v"
```

Where `heur_sol.res` is the initial heuristic solution which was computed before-hand
and supplied using the `--start-with` option, and `datfile.dat` is the datfile corresponding
to the supplied dataset.


### AAAI 2020 paper (∆BIC analysis)

#### BN-SLIM(ETLd) vs ETLd

| ∆BIC               | tw 2 | tw 5 | tw 8 |
|--------------------|-----:|-----:|-----:|
| extremely positive | 89   | 77   | 75   |
| strongly positive  | 3    | 2    | 4    |
| positive           | 0    | 4    | 3    |
| neutral            | 5    | 14   | 15   |

#### BN-SLIM(ETLp) vs ETLp

| ∆BIC               | tw 2 | tw 5 | tw 8 |
|--------------------|-----:|-----:|-----:|
| extremely positive | 91   | 80   | 83   |
| strongly positive  | 0    | 2    | 5    |
| positive           | 0    | 1    | 1    |
| neutral            | 6    | 14   | 8    |

#### BN-SLIM(k-MAX) vs k-MAX

| ∆BIC               | tw 2 | tw 5 | tw 8 |
|--------------------|-----:|-----:|-----:|
| extremely positive | 94   | 74   | 79   |
| strongly positive  | 0    | 1    | 0    |
| positive           | 0    | 1    | 1    |
| neutral            | 0    | 6    | 8    |
| negative           | 0    | 0    | 0    |
| strongly negative  | 0    | 1    | 0    |
| extremely negative | 0    | 11   | 6    |

Above tables obtained by running with the following configuration:

```sh
python -O slim.py <dataset> <treewidth> -r<seed> -t1800 -u kmax -vx --start-with=<sol.res>
```

Where `sol.res` is the initial heuristic solution which was computed before-hand
and supplied using the `--start-with` option.

If that doesn't work, try running with a `aaai` tagged commit.


[1]: https://maxsat-evaluations.github.io/2019/descriptions.html
[2]: https://ipg.idsia.ch/software/blip
[3]: https://github.com/marcobb8/et-learn/archive/master.zip
[4]: https://github.com/marcobb8/et-learn
[5]: https://github.com/marcobb8/et-learn/blob/master/README.md
[6]: https://www.saltycrane.com/blog/2009/05/notes-using-pip-and-virtualenv-django/
[7]: https://stackoverflow.com/a/1534343/1614140
[8]: https://github.com/radum2275/merlin/tree/008d910307a2043f9446676f28010079ea49ec15

