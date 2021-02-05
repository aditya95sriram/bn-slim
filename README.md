# Turbocharging Treewidth-Bounded Bayesian Network Structure Learning

> Instructions provided below have been tested on linux
> Last updated: 5th February 2021


## Required programming languages and software

* `Java SDK` (required to run BLIP)
* `Python 3.6` or higher (required to run BN-SLIM)
* `git` (required to obtain ETL source code)
* `Python 2.7` (required to run ETL)

> Before you start, move the following files into a directory called `slim`:
  (See Section: Directory Structure)
  `blip.py  encoding.py  samer_veith.py  slim.py  utils.py`


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

2. Apply patch `blip.patch` to enable elim-order output

    ```sh
    pushd blip-publish
    patch -ub -p1 -i ../blip.patch
    ```

3. Compile BLIP

    ```sh
    ./compile-blip.sh
    popd
    ```

4. Place jar file in `solvers` directory

    ```sh
    cp blip-publish/blip.jar . -v
    popd
    ```

5. (Optional) Test and check if `tmp.res` contains `elim-order` field after running

    ```sh
    java -jar solvers/blip.jar solver.kmax -j test.jkl -r tmp.res -w 5 -v 1
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

Unzip the files `datasets.zip` and `experiments.zip`


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
├── download-etlearn.sh
├── test.cnf
├── test.jkl
├── test.data
├── demo.res
├── requirements.txt
├── optional_requirements.txt
├── solvers
│   ├── uwrmaxsat.patch
│   ├── blip.patch
│   ├── UWrMaxSat-1.0*
│   ├── uwrmaxsat*
│   ├── blip-publish*
|   ├── blip.jar*
|   └── et_learn*
|       ├── hcet.py
│       └── ... 
├── datasets*
|   ├── dat/
|   └── jkl/
├── experiments*
|   ├── data.csv
|   ├── log_likelihood.csv
│   └── output/
└── slim
    ├── blip.py
    ├── encoding.py
    ├── samer_veith.py
    ├── slim.py
    └── utils.py
```


## Running BN-SLIM

```sh
cd slim
python slim.py ../datasets/jkl/accidents.test.jkl 5 -v -u kmax -t 60
```

> Run `python slim.py --help` for description of available options

> For larger networks it is recommended to invoke as `python -O slim.py ...`


## `.res` files

A `.res` file contains the DAG expressed by means of the parent sets
as well as a field indicating the score of the DAG and optionally
the elimination ordering used to bound the treewidth of the DAG. 
> See `demo.res` for an example.

The patch applied to BLIP allows it to output `.res` files with the 
elimination ordering which isn't supported by it out-of-the-box.

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

### ∆BIC analysis

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

`experiments/data.csv` contains all the data for the 97 datasets from `datasets.zip`, 
 three treewidth values (2, 5, 8) and three random seed values (1,2,3) 
 run with the following configuration:

```sh
python -O slim.py <dataset> <treewidth> -r<seed> -t1800 -u kmax -vx --start-with=<sol.res>
```

Where `sol.res` is the initial heuristic solution which was computed before-hand
and supplied using the `--start-with` option.


### Log-likelihood analysis
The partial data of the log-likelihood analysis is available in 
`experiments/log_likelihood.csv`. A `-1` in the `seed` column indicates 
no seeding (for the ETL algorithms). The `time` column indicates the time
required (in minutes) to generate the network in question, which, 
for the ETL algorithm is always taken as 30 minutes.


[1]: https://maxsat-evaluations.github.io/2019/descriptions.html
[2]: https://ipg.idsia.ch/software/blip
[3]: https://github.com/marcobb8/et-learn/archive/master.zip
[4]: https://github.com/marcobb8/et-learn
[5]: https://github.com/marcobb8/et-learn/blob/master/README.md
[6]: https://www.saltycrane.com/blog/2009/05/notes-using-pip-and-virtualenv-django/
[7]: https://stackoverflow.com/a/1534343/1614140
