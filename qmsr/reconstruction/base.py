"""Base routines for reconstruction map creation"""
from absl import flags
from data_io import ShiftedSVD
import jax.numpy as jnp
import reconstruction.manifold_based as mb
from reconstruction.sampling import sampler

RECONSTRUCTION_STYLE = flags.DEFINE_string(
    'reconstruction_style',
    'sparse',
    'The reconstruction style to use',
)


def get_reconstruction_map(shifted_svd: ShiftedSVD, train_data):
  recmap_constructor = reconstruction_dict[RECONSTRUCTION_STYLE.value]
  reconstruction_map, idcs = recmap_constructor(shifted_svd, train_data)
  assert len(idcs) == flags.FLAGS.n_sample_points
  return reconstruction_map, idcs


def get_sparse_reconstructor(shifted_svd: ShiftedSVD, train_data=None):
  sampling_args = {'full_dim': shifted_svd.U.shape[0], 'train_data': train_data}
  idcs = sampler()(sampling_args)

  def reconstructor(sampled_data_point):
    rec = jnp.zeros((sampling_args['full_dim'], sampled_data_point.shape[1]))
    rec = rec.at[idcs, :].set(sampled_data_point)
    return rec

  return reconstructor, idcs


reconstruction_dict = {
    'sparse': get_sparse_reconstructor,
    'linear': mb.linear_reconstructor,
    'qmsr': mb.get_qm_reconstructor,
    # 'qmgreedylst': mb.greedy_qm_based,
    # 'greedy2': mb.greedy_qm_based2,
}
