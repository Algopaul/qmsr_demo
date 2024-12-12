"""
Vlasov experiments
1. First compare relative errors deim/lst/qmsr (nongreedy/greedy)
2. Compare reduced dimension + sample points
3. Then compute reconstructions
4. Sample points should be reduced_order*{1, 2, 3}
"""
import jax.numpy as jnp
import exp

manidims = list(jnp.arange(1, 16).astype('int32'))
dicts = []

base_dict = {
    'tag': ['vlasov_rel_errs'],
    'data_file': ['data/datafiles/vlasov.h5'],
    'use_svd_file': [True],
    'n_sample_factor': [1, 2, 3, 4],
    'sampling_strategy': ['qdeim'],
    'manidim': manidims,
}

method_combinations = [
    {
        'reconstruction_style': ['linear'],
        'qdeim_inflated_basis': [True, False],
        'greedy_col_idx': [False],
        'reduction_style': ['deim'],
    },
    {
        'reconstruction_style': ['qmsr'],
        'greedy_col_idx': [True],
        'qdeim_inflated_basis': [True, False],
        'reduction_style': ['deim_iter'],
        'encoder_iterations': [0, 20],
        'gauss_newton_damping': [1e-8]
    },
    {
        'reconstruction_style': ['qmsr'],
        'greedy_col_idx': [True, False],
        'reduction_style': ['full'],
        'qdeim_inflated_basis': [False],
        'sampling_strategy': ['linspace'],
    },
    {
        'reconstruction_style': ['linear'],
        'reduction_style': ['full'],
        'qdeim_inflated_basis': [False],
        'sampling_strategy': ['linspace'],
        'greedy_col_idx': [False],
    },
]

for m in method_combinations:
  new_dicts = exp.dict_cartesian_product(**{**base_dict, **m})
  dicts = [*dicts, *list(new_dicts)]

# Reconstructions: r=10, n_sample_factors {1,2}
rec_base_dict = {
    **base_dict,
    'manidim': [10],
    'n_sample_factor': [1, 2],
    'reconstruction_indices': ["[0, 208, 312, 416, 624]"],
    'store_reconstructions': [True],
    'compute_reconstruction_errors': [False],
}
recdicts = []
for m in method_combinations:
  new_dicts = exp.dict_cartesian_product(**{**rec_base_dict, **m})
  recdicts = [*recdicts, *list(new_dicts)]

recfile = "data/reconstructions/vlasov/{}_{}_{}x_g{}_b{}_{}.h5"

for d in recdicts:
  d['reconstruction_outfile'] = recfile.format(
      d['reconstruction_style'],
      d['manidim'],
      d['n_sample_factor'],
      d['greedy_col_idx'],
      d['qdeim_inflated_basis'],
      d['reduction_style'],
  )

dicts = [*dicts, *recdicts]

for d in dicts:
  d['n_sample_points'] = d['manidim'] * d['n_sample_factor']
  d['manidim'] = int(d['manidim'])
  d['n_sample_points'] = int(d['n_sample_points'])
  del d['n_sample_factor']
  if d['reduction_style'] == 'full':
    d['n_sample_points'] = 360000
  if ((d['manidim'] == 1) and (d['n_sample_points'] > 1) and
      (d['sampling_strategy'] == 'qdeim') and
      (d['qdeim_inflated_basis'] == False)):
    # do nothing because this will error
    print('Not adding this')
  else:
    exp.add_config(d)
