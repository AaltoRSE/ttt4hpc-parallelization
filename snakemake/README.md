# Demo: Parameter sweep in parallel using an array job

In this demo / exercise session, we run a K-nearest neighbors classifier on the classic [Iris data set](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) with different parameter combinations and plot the corresponding decision boundaries. We submit this task to the [Aalto Triton cluster](https://scicomp.aalto.fi/triton/) using an array job.

The demo is based on the [Nearest Neighbor Classification example of the scikit-learn toolkit](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html).


## Run on Aalto Triton cluster

Clone repo to work directory
```
git clone https://github.com/AaltoRSE/ttt4hpc-snakemake-demo $WRKDIR/ttt4hpc-snakemake-demo
```

Change directory

```
cd $WRKDIR/ttt4hpc-snakemake-demo/arrayjob
```

Create a conda environment

```
module load miniconda
mamba env create -f iris_knn.yml -p env/
```

Submit job

```
sbatch submit_arrayjob.sh
```


### Parallelization with Snakemake

Change directory

```
cd $WRKDIR/ttt4hpc-snakemake-demo/snakemake
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


## Comparing the array jobs and Snakemake

DRAFT

The same essential information needs to be specified in both cases (requested computational resources, environments, ) but the structuring is different: Slurm submission scripts versus Snakefiles and Snakemake profiles.

However, the approaches have their own advantages:

- Snakemake automatically checks existing result / intermediate files and does not recreate them
- Array job can be launced with a single submission file which can be nice
- 

## Running on different clusters 

DRAFT

Array job (or Slurm in general) and Snakemake specifications are, in principle, the same in different infrastructures. Nevertheless, different infras usually have their own peculiarities. For example, on the Aalto Triton, we can use conda/mamba directly to create environments. Meanwhile, on CSC's Puhti, we can not use conda/mamba directly but via Tykky. This demands changes to both Slurm submission scripts and Snakefiles used by Snakemake.  






