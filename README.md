# TTT4HPC: Parallelization

## Demo: From Jupyter Notebook to a Slurm array job

In this session, we examine a Jupyter notebook which trains [K nearest neighbor (knn) classifiers]() on the [Iris data set](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) and plots their respective decision boundaries on the test set.
We train the classifier using different number of neighbors, that is, we make a parameter sweep. Since the classifiers corresponding to different neighbor values do not depend on each other, we can train them in parallel. To this end, we will go through the necessary steps to run the notebook on a cluster using an Slurm array job. The steps are:

- Convert the Jupyter notebook to a Python script
- Refactor the Python script to three scripts
  - One to preprocess data
  - One to create a parameter array file (TODO: maybe just incorporate this into fit and plot script)
  - One to train a knn and plot decision boundaries for each parameter in the array
- Create an array job submission script

Finally, we preprocess the data, create a parameter array, and submit the array job on [Aalto Triton cluster](https://scicomp.aalto.fi/triton/)

The notebook is based on the [Nearest Neighbor Classification example of the scikit-learn toolkit](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html).

### Clone the repo

Clone the repo
```
git clone https://github.com/AaltoRSE/ttt4hpc-parallel ttt4hpc-parallel
```


### Examine Jupyter notebook

TODO: Description

`iris_knn.ipynb`


### Convert notebook to Python script

TODO: Description

`arrayjob/iris_knn.py`


### Refactor Python script to multiple scripts

TODO: Description

`arrayjob/src/preprocess_data.py` # Needs to be run once

`arrayjob/src/create_parameter_array.py` # Needs to be run once (TODO: incorporate to `fit_plot.py`)

`arrayjob/src/fit_plot.py` # Multiple runs corresponding to different parameters, can be run in parallel


### Create array job submission script

TODO: Description

`arrayjob/submit_arrayjob.sh`


### Run on Aalto Triton cluster

TODO: Description

Clone repo to work directory
```
git clone https://github.com/AaltoRSE/ttt4hpc-parallel $WRKDIR/ttt4hpc-parallel
```

Change directory

```
cd $WRKDIR/ttt4hpc-parallel/arrayjob
```

Create a Conda environment

```
module load miniconda
mamba env create -f knn_iris.yml -p env/
```

>_**NOTE:**_ Different clusters can have varying practices concerning Conda environments. On Aalto Triton, you can use [Conda/Mamba directly](https://scicomp.aalto.fi/triton/apps/python-conda/). On CSC's Puhti/Mahti/LUMI, you need to use their [Tykky wrapper tool](https://docs.csc.fi/computing/containers/tykky/). 

Preprocess data and show the preprocessed data file

```
python3 src/preprocess_data.py
ls -l data/preprocessed
```

Create parameter array and show the created jsonlines file

```
python3 src/create_parameter_array.py
ls -l params/
cat params/params.jsonl
```

Run `src/fit_plot.py` corresponding to all parameters in `params/params.jsonl` as an array job

```
sbatch submit_arrayjob.sh
```


### Exercises


#### Exercise

Instead of looping over the number of neighbors (`n_neighbors`) and keeping the distance metric (`metric`) fixed, loop over all combinations of `n_neighbors` and `metric`, where `n_neighbors` takes values from 1, 2, 4, 8, 16, 32, and 64, and `metric` takes values "cityblock", "cosine", "euclidean", "haversine", "l1", "l2", "manhattan", and "nan_euclidean".

Hint: Modify `arrayjob/src/create_parameter_array.py` and `submit_arrayjob.sh`
FIXME: Modify `arrayjob/src/fit_plot.py` and `submit_arrayjob.sh`

#### Exercise (optional)

????



## Demo: Using Snakemake

In this session, we use Snakemake to automatize the workflow consisting of the Python scripts

- `preprocess_data.py`
- `create_parameters.py`
- `fit_plot.py`

The necessary steps are

- restructure the project according to Snakemake's guideline
- write a Snakefile specifying the inputs and outputs of each script
- write a profile file specifying the resources requested from Slurm

### Restructure the project

TODO: Describe project structure with empty placeholder Snakefile and profile file


### Write the Snakefile

TODO: describe 



### Write a profile file

TODO: describe


### Run on Aalto Triton

Change directory

```
cd $WRKDIR/ttt4hpc-parallel/snakemake
```

Create a conda environment for Snakemake and Snakemake Slurm executor plugin with

```
module load miniconda
mamba env create --file snakemake.yml --prefix env/
```

Run Snakemake with

```
snakemake --snakefile workflow/Snakefile --profile profiles/slurm --use-conda
```


What happens:

1. Snakemake infers from `workflow/Snakefile` that the required input files specified in rule "All" can be created using the rule "plot_decision_boundaries" in an embarassingly parallel manner. (Note that input files of the rule "All" are our target output files.)

2. The profile configuration specified in `profiles/slurm/config.yaml` tells Snakemake to submit the jobs to Slurm and to request the specified resources (cpus, memory, runtime, etc.). The resources can be specified for each rule individually.

3. The option `--use-conda` tells Snakemake to look for Conda/Mamba environments in `.snakemake/conda/` for the rule "plot_decision_boundaries". These environments will be created if they do not exist yet.











