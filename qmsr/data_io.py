"""Load input data and store results"""
import h5py
from absl import flags
from absl import logging
import jax.numpy as jnp
from collections import namedtuple
import os
import time

ShiftedSVD = namedtuple('ShiftedSVD', ['U', 'S', 'VT', 'shift'])

USE_SVD_FILE = flags.DEFINE_boolean(
    'use_svd_file',
    False,
    'Whether to use SVD file for computing the manifold',
)
SVD_FILE = flags.DEFINE_string(
    'svd_file',
    '',
    'Data file for SVD, (hdf5-file with fields `U, S, V, shift`)',
)
DATA_FILE = flags.DEFINE_string(
    'data_file',
    '',
    'Data file for manifold data,'
    '(hdf5-file with fields `train_data, val_data, test_data`)',
)
SHIFT_METHOD = flags.DEFINE_string(
    'shift_method',
    'mean',
    'Method for shifting data `mean, zero, first`'
    '(only relevant when not using svd file)',
)
STORE_SVD_FILE = flags.DEFINE_boolean(
    'store_svd_file',
    True,
    'Whether to store the computed svd',
)


def get_svd_filename():
  if len(SVD_FILE.value) > 0:
    return SVD_FILE.value
  else:
    base_name = os.path.basename(DATA_FILE.value)
    return os.path.join(
        'data/datafiles/svd_files',
        SHIFT_METHOD.value,
        base_name,
    )


def load_shifted_svd(svd_filename=None):
  if svd_filename is None:
    svd_filename = get_svd_filename()
  with h5py.File(svd_filename) as file:
    lsvecs = jnp.asarray(file['U'])
    svals = jnp.asarray(file['S'])
    rsvecs = jnp.asarray(file['V']).T
    if 'shift' in file.keys():
      shift = jnp.asarray(file['shift'])
    else:
      shift = jnp.zeros(lsvecs.shape[0])
  return ShiftedSVD(lsvecs, svals, rsvecs, shift)


def store_shifted_svd(filename: str, svd: ShiftedSVD, compute_time=None):
  with h5py.File(filename, 'w') as f:
    f.create_dataset('U', data=svd.U)
    f.create_dataset('S', data=svd.S)
    f.create_dataset('V', data=svd.VT.T)
    f.create_dataset('shift', data=svd.shift)
    f.create_dataset('compute_time', data=compute_time)
  pass


def get_shift(data, shift_method=None):
  if shift_method is None:
    shift_method = SHIFT_METHOD.value
  if shift_method == 'mean':
    shift = jnp.mean(data, axis=1)
  elif shift_method == 'zero':
    shift = jnp.zeros(data.shape[0])
  elif shift_method == 'first':
    shift = data[:, 0]
  else:
    raise ValueError('Unknown shift method')

  return shift


def compute_shifted_svd(data_filename=None, shift_method=None):
  if data_filename is None:
    data_filename = DATA_FILE.value
  if shift_method is None:
    shift_method = SHIFT_METHOD.value
  f = get_datafile(data_filename)
  train_data = jnp.array(f['train_data'])
  f.close()
  shift = get_shift(train_data, shift_method)
  t0 = time.time()
  computed_svd = jnp.linalg.svd(
      train_data - shift[:, None], full_matrices=False)
  t1 = time.time()
  elapsed_time = t1 - t0
  shifted_svd = ShiftedSVD(
      computed_svd[0],
      computed_svd[1],
      computed_svd[2],
      shift,
  )
  if STORE_SVD_FILE.value:
    filename = get_svd_filename()
    store_shifted_svd(filename, shifted_svd, elapsed_time)
  return shifted_svd


def get_datafile(data_filename=None, mode='r'):
  if data_filename is None:
    data_filename = DATA_FILE.value

  data_dir = os.path.dirname(data_filename)

  if not os.path.exists(data_dir) and data_dir:
    os.makedirs(data_dir)
    logging.info('Created directory: %s', data_dir)
  return h5py.File(data_filename, mode)


def should_recompute_svd(data_filename: str, svd_filename: str) -> bool:
  """Check if the SVD should be recomputed based on file modification times."""
  if not os.path.isfile(svd_filename):
    logging.warn('SVD file %s does not exist. Computing SVD.', svd_filename)
    return True

  svd_mtime = os.path.getmtime(svd_filename)
  data_mtime = os.path.getmtime(data_filename)

  if data_mtime > svd_mtime:
    logging.warn('Data file %s is newer than SVD file %s. Recomputing SVD.',
                 data_filename, svd_filename)
    return True

  return False


def load_or_recompute_svd(svd_filename: str, data_filename: str) -> ShiftedSVD:
  """Load the SVD from file or recompute it if necessary."""
  if should_recompute_svd(data_filename, svd_filename):
    return compute_shifted_svd()

  logging.warn('Loading from SVD file %s.', svd_filename)
  return load_shifted_svd()


def get_svd() -> ShiftedSVD:
  """Main function to get the SVD, either by loading from file or recomputing it."""
  svd_filename = get_svd_filename()
  data_filename = DATA_FILE.value

  if USE_SVD_FILE.value:
    return load_or_recompute_svd(svd_filename, data_filename)

  return compute_shifted_svd()
