## Empirical sparse regression on quadratic manifolds

Code for recreating the experiments in
```tex
@article{SchwerdtnerGP2024Empirical,
    authors = {P. Schwerdtner and S. Gugercin and B. Peherstorfer},
    title = {Empirical sparse regression on quadratic manifolds},
    year = {2024}
}
```

### Installation

1. Download this repo `git clone https://github.com/Algopaul/qmsr_demo.git`
2. In the repository, run `bash configure.sh`
3. run `make install`

### First steps

To get an understanding of the method, you can run `make vlasov_example` to compute a sparse quadratic manifold reconstruction of the Vlasov example we present in our paper.

To rerun our experiments you can run `make run.sh`; investigate and run the commands within `run.sh` and then run `make analysis`, which generates tables in `data/results_tables`. We report the errors in these tables in our paper.
