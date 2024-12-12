create table if not exists results (
  hash text primary key,
  tag text,
  -- Loading fields
  use_svd_file boolean,
  svd_file text,
  data_file text,
  shift_method text,
  -- Config fields
  n_sample_points integer,
  reconstruction_style text,
  reduction_style text,
  manidim integer,
  greedy_col_idx integer,
  greedy_nconsider integer,
  lstsq_regmag double precision,
  greedy_idx_in text,
  sampling_strategy text,
  split_lstsq boolean,
  reconstruction_lstsq_regmag double precision,
  feature_map text,
  qdeim_inflated_basis boolean,
  track_deim_constant boolean,
  gauss_newton_damping double precision,
  encoder_iterations integer,
  -- IO fields
  compute_reconstruction_errors boolean default true,
  store_reconstructions boolean default false,
  reconstruction_outfile text,
  reconstruction_indices text,
  batch_size_gb float,
  -- Results fields
  train_err double precision,
  test_err double precision,
  deim_constant double precision,
  -- Status fields
  git_hash text,
  scheduled boolean DEFAULT false,
  running boolean DEFAULT false,
  failed boolean DEFAULT false,
  completed boolean DEFAULT false
);
