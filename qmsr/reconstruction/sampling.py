from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
import exp

N_SAMPLE_POINTS = flags.DEFINE_integer(
    'n_sample_points',
    10,
    'The number of points that the reconstruction_map'
    'gets for reconstruction of the full snapshot',
)
SAMPLING_STRATEGY = flags.DEFINE_string(
    'sampling_strategy',
    'linspace',
    'The sample point selection strategy',
)
TRACK_DEIM_CONSTANT = flags.DEFINE_boolean(
    'track_deim_constant',
    False,
    'Whether to track the deim constant',
)


def sampler():
  return sampling_strategies[SAMPLING_STRATEGY.value]


def linspace_samples(sampling_args, n_sample_points=None):
  if n_sample_points is None:
    n_sample_points = N_SAMPLE_POINTS.value
  idcs = jnp.linspace(0, sampling_args['full_dim'] - 1, n_sample_points)
  return jnp.floor(idcs + 1e-12).astype('int')


def random_samples(sampling_args, n_sample_points=None):
  if n_sample_points is None:
    n_sample_points = N_SAMPLE_POINTS.value
  idcs = (sampling_args['full_dim'] - 1) * jax.random.uniform(
      jax.random.key(0), (n_sample_points,))
  return jnp.sort(jnp.floor(idcs).astype('int'))


def deim_constant(basis, sample_points):
  _, S, _ = jnp.linalg.svd(basis[sample_points, :], full_matrices=False)
  return 1 / S[-1]


def qdeim_samples(sampling_args, n_sample_points=None):
  if n_sample_points is None:
    n_sample_points = N_SAMPLE_POINTS.value
  sample_points = jnp.sort(
      oversampled_qdeim_points(sampling_args['basis'], n_sample_points))
  return sample_points


def oversampled_qdeim_points(U, m):
  p = qdeim_points(U, min(m, U.shape[1]))
  if (len(p) > 1) and (m > U.shape[1]):
    logging.info('Oversampling DEIM')
    p = greedy_extra_search(U, p, m)
  return jnp.sort(p)


def qdeim_points(U, m):
  if m > U.shape[1]:
    logging.error('Requested more sampling points than basis vectors.'
                  'Use sampling method `oversampled_qdeim_points` instead')
  logging.info('Basis has shape %s', U.shape)
  _, P = scipy.linalg.qr(U.T, pivoting=True, mode='r')
  logging.info('All points are %s', P)
  logging.info('Restricting to the first %s', m)
  return jnp.sort(P[:m])


def greedy_extra_search(U, p, m):
  """
  Extra search algorithm from

  Peherstorfer, Benjamin and Drma\v{c}, Zlatko and Gugercin, Serkan:
  *Stability of Discrete Empirical Interpolation and Gappy Proper Orthogonal
  Decomposition with Randomized and Deterministic Sampling Points*
  SIAM Journal on Scientific Computing 42(5) A2837-A2864, 2020
  www.doi.org/10.1137/19M1307391
  """
  for _ in range(len(p), m):
    _, S, W = jnp.linalg.svd(U[p, :])
    g = S[-2]**2 - S[-1]**2
    Ub = W @ U.T
    Ubss = jnp.sum(Ub**2, axis=0)
    r = g + Ubss
    r = r - jnp.sqrt((g + Ubss)**2 - 4 * g * Ub[-1, :]**2)
    I = jnp.argsort(-r)
    for i in I:
      if i not in p:
        p = jnp.hstack((p, jnp.array([i])))
        break
  assert len(p) == m
  return p


def greedy_samples(sampling_args, n_sample_points=None):
  if n_sample_points is None:
    n_sample_points = N_SAMPLE_POINTS.value
  basis = sampling_args['basis']
  p = jnp.asarray([10, 20])
  return jnp.sort(greedy_extra_search(basis, p, n_sample_points))


def lst_greedy_samples(sampling_args, n_sample_points=None):
  """
  Experimental sampling algorithm (does not work well)
  """
  if n_sample_points is None:
    n_sample_points = N_SAMPLE_POINTS.value
  td = sampling_args['train_data']
  logging.info('Train data shape %s', td.shape)
  basis = sampling_args['basis']
  tdTbasis = sampling_args['tdTbasis']
  idcs = jnp.zeros(0).astype('int')
  logging.info(jnp.linalg.norm(tdTbasis))
  logging.info(jnp.linalg.norm(td))

  idcs = oversampled_qdeim_points(td, n_sample_points)

  fig = plt.figure()
  plt.plot(tdTbasis)
  plt.savefig('red_coords.png', dpi=300)
  plt.close(fig)
  # exit(1)

  for i in range(n_sample_points):
    normvals = jnp.zeros(basis.shape[0])
    logging.info("Current idcs=%s", idcs)

    def false_fun(idcs_test):
      _, resid, _, _ = jnp.linalg.lstsq(
          td[idcs_test, :].T,
          tdTbasis,
          rcond=1e-8,
      )
      return jnp.linalg.norm(resid)

    def true_fun(idcs_test):
      return jnp.float32(jnp.inf)

    def body_fun(j, normvals):
      idcs_test = jnp.hstack((idcs, j))
      A = td[idcs_test, :].T
      normval = jax.lax.cond(
          jnp.linalg.norm(A) < 1.0e-8,
          true_fun,
          false_fun,
          idcs_test,
      )
      normvals = normvals.at[j].set(normval)
      return normvals

    # normvals = jax.lax.fori_loop(0, basis.shape[0], body_fun, normvals)

    logging.info("Min norm %s", jnp.min(normvals))
    logging.info("Max norm %s", jnp.max(normvals))
    # idcs = jnp.asarray((*idcs, jnp.argmin(normvals)))

    fig = plt.figure()
    sol = jnp.linalg.lstsq((td[idcs, :]).T, tdTbasis, rcond=1e-8)[0]
    plt.plot(td[idcs, :].T @ sol)
    plt.savefig(f'app_red_coords_{i}.png', dpi=300)
    plt.close(fig)

    fig = plt.figure()
    shape = (600, 600)
    normvals_reshape = jnp.reshape(normvals, shape)
    plt.imshow(normvals_reshape)
    plt.colorbar()
    rows, cols = jnp.unravel_index(idcs, normvals_reshape.shape)
    plt.savefig(f'normvals_{i}.png', dpi=400)
    plt.scatter(cols, rows, color='red')
    plt.savefig(f'scatter_normvals_{i}.png', dpi=400)
    plt.close(fig)

  return idcs


sampling_strategies = {
    'linspace': linspace_samples,
    'random': random_samples,
    'qdeim': qdeim_samples,  # Uses oversampled qdeim by default
    'greedy': greedy_samples,
    'lst_greedy': lst_greedy_samples,
}
