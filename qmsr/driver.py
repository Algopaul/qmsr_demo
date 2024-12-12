"""Driver for QM-DEIM experiments"""
import jax
import jax.numpy as jnp
from absl import flags, logging
from absl import app
from data_io import get_svd, get_datafile
from reconstruction.base import get_reconstruction_map
from error_computation import chunkwise_rel_error
import exp

jax.config.update('jax_enable_x64', True)

HASH = flags.DEFINE_string('hash', '', 'The hash of the experiment')
TAG = flags.DEFINE_string('tag', 'default', 'The tag of the experiment group')
TRACK_EXPERIMENT = flags.DEFINE_boolean(
    'track_experiment',
    True,
    'Whether to track the experiment',
)
COMPUTE_RECONSTRUCTION_ERRORS = flags.DEFINE_boolean(
    'compute_reconstruction_errors',
    True,
    'Whether to compute the reconstruction errors',
)
STORE_RECONSTRUCTIONS = flags.DEFINE_boolean(
    'store_reconstructions',
    False,
    'Whether to store reconstructions of given indices',
)
RECONSTRUCTION_OUTFILE = flags.DEFINE_string(
    'reconstruction_outfile',
    'data/reconstruction.h5',
    'The directory in which the reconstructions are stored.',
)
RECONSTRUCTION_INDICES = flags.DEFINE_multi_integer(
    'reconstruction_indices',
    [0],
    'The indices from the test_data field for'
    'which reconstructions should be stored.',
)


def get_errors(wrapped_map):
  datafile = get_datafile()
  errs = {}
  for field in ['train', 'test']:
    try:
      errs[field + '_err'] = chunkwise_rel_error(
          wrapped_map,
          datafile[field + '_data'],
      )
    except Exception as e:
      logging.info(e)
      logging.info('Error computation failed for field %s', field)

  datafile.close()
  return errs


def get_wrapped_map(reconstruction_map, idcs):

  def wrapped_map(data):
    return reconstruction_map(data[idcs, :])

  return wrapped_map


def store_reconstructions(wrapped_map, row_idcs):
  logging.info(
      'Computing reconstructions in test_data at %s',
      RECONSTRUCTION_INDICES.value,
  )
  df = get_datafile()
  data_field = df['test_data']
  rec_idcs = jnp.asarray(RECONSTRUCTION_INDICES.value)
  true_data = jnp.asarray(data_field[:, rec_idcs])  # pyright: ignore
  data = true_data
  reconstructed_data = wrapped_map(data)
  error = reconstructed_data - true_data
  df.close()
  df_out = get_datafile(RECONSTRUCTION_OUTFILE.value, 'w')
  fields = [
      'reconstructed_data',
      'error',
      'given_data',
      'corrupted_data',
      'idcs',
      'row_idcs',
  ]
  data = [reconstructed_data, error, true_data, data, rec_idcs, row_idcs]
  for field, data in zip(fields, data):
    df_out.create_dataset(field, data=data)
  df_out.close()
  logging.info('Storing at %s', RECONSTRUCTION_OUTFILE.value)
  pass


def main(_):
  if TRACK_EXPERIMENT.value:
    exp.mark_running(HASH.value)
  try:
    fields_to_store = exp.config_fields('config_flags.cfg')[1:]
    results = exp.flags_to_dict(flags.FLAGS, fields_to_store)
    logging.info('Loading train data')
    shifted_svd = get_svd()
    df = get_datafile()
    train_data = df['train_data']
    logging.info('Computing reconstruction map')
    reconstruction_map, row_idcs = get_reconstruction_map(
        shifted_svd,
        train_data,
    )
    assert len(row_idcs) == flags.FLAGS.n_sample_points
    wrapped_map = get_wrapped_map(reconstruction_map, row_idcs)

    logging.info('Starting evaluation')
    if COMPUTE_RECONSTRUCTION_ERRORS.value:
      errs = get_errors(wrapped_map)
      logging.info('Computed errors: %s', errs)
      results = {**results, **errs}

    if STORE_RECONSTRUCTIONS.value:
      logging.info('Storing reconstructions')
      store_reconstructions(wrapped_map, row_idcs)

    if TRACK_EXPERIMENT.value:
      exp.add_result(HASH.value, results)
      exp.mark_completed(HASH.value)
  except Exception as e:
    exp.mark_failed(HASH.value)
    raise e

  pass


if __name__ == '__main__':
  app.run(main)
