include ./config/runner_definition.mk	

dirs=data data/datafiles $(addprefix data/datafiles/svd_files/,mean zero first)

$(dirs): %:
	mkdir -p ${*}

.venv:
	$(DEFAULTPYTHON) -m venv .venv
	.venv/bin/pip install -e .
	.venv/bin/pip install -e .[dev]

install: .venv $(dirs) data/db.sqlite3
	.venv/bin/pip install -e .
	.venv/bin/pip install -e .[dev]

data/db.sqlite3: initdb.sql | data
	sqlite3 data/db.sqlite3 < initdb.sql

data/datafiles/vlasov.h5: | data/datafiles
	$(RUN) .venv/bin/python ./qmsr/benchmark/vlasov.py --ode_t1=5.0 --storing=True

benchmark_trajectories=$(addprefix data/datafiles/,vlasov.h5)

benchmark_data: $(benchmark_trajectories)

vlasov_example: | $(dirs) data/datafiles/vlasov.h5
	$(RUN) .venv/bin/python ./qmsr/driver.py\
		--track_experiment=False\
		--data_file=./data/datafiles/vlasov.h5\
		--use_svd_file=True\
		--shift_method=mean\
		--reconstruction_style=qmsr\
		--reduction_style=deim_iter\
		--manidim=10\
		--n_sample_points=10\
		--sampling_strategy=qdeim\
		--greedy_col_idx=True\
		--greedy_nconsider=200\
		--qdeim_inflated_basis=True

.PHONY: experiments
experiments: data/datafiles/vlasov.h5 data/db.sqlite3
	.venv/bin/python experiments/vlasov.py


analysis:
	.venv/bin/python experiments/vlasov_analysis.py


run.sh: experiments
	.venv/bin/python experiments/create_runner.py
