"""Error computation in chunks"""
from jax.random import key
import jax.numpy as jnp
from absl import logging, flags
import gc

BATCH_SIZE_GB = flags.DEFINE_float(
    'batch_size_gb',
    3,
    'The batch size in GB when doing the evalutation',
)


def get_batch_size(n):
  gb_per_batch = BATCH_SIZE_GB.value
  gb_per_col = n * 8 * 1e-9  # assuming double precision
  return int(gb_per_batch // gb_per_col)


def chunkwise_rel_error(reconstruction_map, h5_field, batch_size=None):
  if batch_size is None:
    batch_size = get_batch_size(h5_field.shape[0])
  n_batches = int(jnp.ceil(h5_field.shape[1] / batch_size))
  logging.info('Evaluation: Divided dataset into %i batches of size %i',
               n_batches, min(h5_field.shape[1], batch_size))
  err = 0.0
  norm_data = 0.0
  for i in range(n_batches):
    logging.info('Evaluation: processing chunk %i/%i', i, n_batches)
    col0 = i * batch_size
    batch_start = col0
    batch_end = min(col0 + batch_size, h5_field.shape[1])
    true_data = jnp.asarray(h5_field[:, batch_start:batch_end])
    data = true_data
    reconstructed_data = reconstruction_map(data)
    err += jnp.linalg.norm(reconstructed_data - true_data)**2
    norm_data += jnp.linalg.norm(true_data)**2
    del data, true_data, reconstructed_data
    gc.collect()
  logging.info('Norm of true data is: %s', norm_data)
  logging.info('Norm of error is: %s', err)
  return float(jnp.sqrt(err) / jnp.sqrt(norm_data))
