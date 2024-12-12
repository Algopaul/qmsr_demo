"""Manifold based reconstruction maps"""
from absl import flags
from data_io import ShiftedSVD
import jax.numpy as jnp
import jax
from reconstruction.sampling import sampler
from reconstruction.sampling import deim_constant
from absl import logging
import exp

MANIDIM = flags.DEFINE_integer(
    'manidim',
    10,
    'The manifold dimension',
)
RECONSTRUCTION_LSTSQ_REGMAG = flags.DEFINE_float(
    'reconstruction_lstsq_regmag',
    1e-8,
    'The regularization magnitude when computing'
    'the subsampled reduction matrix',
)
DEIM_LSTSQ_REGMAG = flags.DEFINE_float(
    'deim_lstsq_regmag',
    1e-8,
    'The regularization magnitude when computing'
    'the DEIM reconstruction',
)
GREEDY_COL_IDX = flags.DEFINE_boolean(
    'greedy_col_idx',
    False,
    'Whether to select the columns greedily when constructing the quadratic manifold',
)
GREEDY_NCONSIDER = flags.DEFINE_integer(
    'greedy_nconsider',
    200,
    'The number of cols to consider.',
)
QDEIM_INFLATED_BASIS = flags.DEFINE_boolean(
    'qdeim_inflated_basis',
    False,
    'Whether to use a basis with n_sample_points columns',
)
REDUCTION_STYLE = flags.DEFINE_string(
    'reduction_style',
    'lstsq',
    'How to compute the reduced coordinates from sub-sampled snapshots',
)
SAMPLING_BASIS_SIZE = flags.DEFINE_integer(
    'sampling_basis_size',
    -1,
    'When set to a value larger than zero,'
    'this sets the number of columns to be'
    'used in the basis (typically used for ODEIM)'
    'independent of sampling_basis_size only n_sample_points'
    'will be used.',
)
ENCODER_ITERATIONS = flags.DEFINE_integer(
    'encoder_iterations',
    0,
    'The number of Gauss Newton iterations in the encoder',
)
GAUSS_NEWTON_DAMPING = flags.DEFINE_float(
    'gauss_newton_damping',
    1e-8,
    'damping magnitude of the Gauss Newton iterations',
)
GAUSS_NEWTON_CONVERGENCE_RTOL = flags.DEFINE_float(
    'gauss_newton_convergence_rtol',
    1e-12,
    'relative convergence tolerance for gauss newton iterations',
)


def kron_map(x):
  return jnp.concatenate([x[i] * x[:i + 1] for i in range(x.shape[0])], axis=0)


def get_sampling_basis(shifted_svd):
  logging.info(
      'Computing sample points using strategy: %s',
      flags.FLAGS.sampling_strategy,
  )
  if QDEIM_INFLATED_BASIS.value:
    logging.info("Using an inflated basis")
    if SAMPLING_BASIS_SIZE.value > 0:
      logging.info('Using independent sampling basis size.')
      return shifted_svd.U[:, :SAMPLING_BASIS_SIZE.value]
    else:
      return shifted_svd.U[:, :flags.FLAGS.n_sample_points]
  else:
    if SAMPLING_BASIS_SIZE.value > 0:
      logging.info('Using independent sampling basis size.')
      return shifted_svd.U[:, :SAMPLING_BASIS_SIZE.value]
    else:
      return shifted_svd.U[:, :MANIDIM.value]


def shift_wrapper(rec_fn, shift, idcs):
  """
  Handles shifting of the given data because
  we base our computations on a shifted SVD.
  """
  sampled_shift = shift[idcs]

  def reconstruct(sampled_data_points):
    s_sampled_data_points = sampled_data_points - sampled_shift[:, None]
    s_rec = rec_fn(s_sampled_data_points)
    return s_rec + shift[:, None]

  return reconstruct


def get_deim_reduction(
    basis,
    row_idcs,
    train_data_row_idcs=None,
    full_reduced_coordinates=None,
    nonlinear_coeff=None,
    n_iterations=None,
):
  del full_reduced_coordinates

  def reduced_coordinates(sampled_data_points):
    res, _, _, _ = jnp.linalg.lstsq(
        basis[row_idcs, :],
        sampled_data_points,
        rcond=DEIM_LSTSQ_REGMAG.value,
    )
    return res

  return reduced_coordinates


def get_iterative_deim_reduction(
    basis,
    row_idcs,
    train_data_row_idcs,
    full_reduced_coordinates,
    nonlinear_coeff=None,
    n_iterations=None,
):
  del full_reduced_coordinates
  if n_iterations is None:
    n_iterations = ENCODER_ITERATIONS.value

  if nonlinear_coeff is not None:

    def jac(x):
      return basis[row_idcs] + nonlinear_coeff[row_idcs] @ kron_map_deriv(x)

    def gn_lhs(xr, i):
      return jnp.vstack((
          jac(xr[:, i]),
          GAUSS_NEWTON_DAMPING.value * jnp.eye(xr.shape[0]),
      ))

  def encode(sampled_data_points):
    xr, _, _, _ = jnp.linalg.lstsq(
        basis[row_idcs, :],
        sampled_data_points,
        rcond=DEIM_LSTSQ_REGMAG.value,
    )

    if n_iterations > 0:
      assert nonlinear_coeff is not None

      def rel_err(xrs):
        recpoints = basis[row_idcs] @ xrs - nonlinear_coeff[
            row_idcs] @ kron_map(xrs)
        return jnp.linalg.norm(sampled_data_points - recpoints)

      logging.info('Perform %s Newton iterations', n_iterations)

      def gn_rhs(xr, i):
        res = (basis[row_idcs] @ xr[:, i] + nonlinear_coeff[row_idcs]
               @ kron_map(xr[:, i])) - sampled_data_points[:, i]
        return jnp.hstack((res, jnp.zeros((xr.shape[0]))))

      def body_fun(i, xr):
        updatei = xr[:, i] - jnp.linalg.lstsq(
            gn_lhs(xr, i),
            gn_rhs(xr, i),
        )[0]
        xr = xr.at[:, i].set(updatei)
        return xr

      rel_err_old = rel_err(xr)
      logging.info('Iteration %4i [rel-error] %.4e', 0, rel_err_old)
      for j in range(n_iterations):
        xr = jax.lax.fori_loop(0, xr.shape[1], body_fun, xr)
        rel_err_new = rel_err(xr)
        conv_crit = jnp.abs(rel_err_new - rel_err_old)
        rel_err_old = rel_err_new
        logging.info(
            'Iteration %4i [rel-error] %.4e [conv crit] %.4e',
            j + 1,
            rel_err_new,
            conv_crit,
        )
        if conv_crit < GAUSS_NEWTON_CONVERGENCE_RTOL.value:
          break
    return xr

  return encode


def get_lstsq_reduction(
    basis,
    row_indices,
    train_data_row_idcs,
    full_reduced_coordinates,
    nonlinear_coeff=None,
    n_iterations=None,
):
  del basis
  subsampled_basis = fit_subsampled_basis(
      train_data_row_idcs,
      full_reduced_coordinates,
  )
  subsampled_basisT = subsampled_basis.T

  def reduced_coordinates(sampled_data_points):
    return subsampled_basisT @ sampled_data_points

  return reduced_coordinates


def get_full_reduction(
    basis,
    row_indices,
    train_data_row_idcs,
    full_reduced_coordinates,
    nonlinear_coeff=None,
    n_iterations=None,
):
  del row_indices, full_reduced_coordinates
  basisT = basis.T

  def reduced_coordinates(sampled_data_points):
    return basisT @ sampled_data_points

  return reduced_coordinates


REDUCTION_FUNCTIONS = {
    'lstsq': get_lstsq_reduction,
    'deim': get_deim_reduction,
    'deim_iter': get_iterative_deim_reduction,
    'full': get_full_reduction,
}


def fit_subsampled_basis(
    subsampled_train_data,
    reduced_coordinates,
):
  X, _, _, _ = jnp.linalg.lstsq(
      subsampled_train_data.T,
      reduced_coordinates.T,
      rcond=RECONSTRUCTION_LSTSQ_REGMAG.value,
  )
  return X


def linear_reconstructor(shifted_svd: ShiftedSVD, train_data):
  n_red = MANIDIM.value
  basis = shifted_svd.U[:, :n_red]
  sampling_args = {
      'basis': get_sampling_basis(shifted_svd),
      'full_dim': basis.shape[0],
      'reduced_dim': n_red,
  }
  row_indices = sampler()(sampling_args)
  assert len(row_indices) == flags.FLAGS.n_sample_points
  logging.info('Collected samples %s', row_indices)
  train_data = train_data[row_indices, :] - shifted_svd.shift[row_indices, None]
  if flags.FLAGS.n_sample_points < 100:
    logging.info('Using row indices: %s', row_indices)
  full_reduced_coordinates = (
      shifted_svd.S[:n_red, None] * shifted_svd.VT[:n_red, :])
  reduced_coords_fn = REDUCTION_FUNCTIONS[REDUCTION_STYLE.value](
      basis,
      row_indices,
      train_data,
      full_reduced_coordinates,
  )
  deco = deim_constant(
      basis,
      row_indices,
  )
  logging.info('Deim constant is %s', float(deco))
  exp.add_result(
      flags.FLAGS.hash,
      {
          'deim_constant': float(deco),
      },
  )

  def reconstruct(sampled_data_points):
    reduced_coordinates = reduced_coords_fn(sampled_data_points)
    assert reduced_coordinates.shape[0] == MANIDIM.value
    return basis @ reduced_coordinates

  return shift_wrapper(reconstruct, shifted_svd.shift, row_indices), row_indices


def principal_angles(U, V):
  M = U.T @ V
  _, S, _ = jnp.linalg.svd(M, full_matrices=False)
  S = jnp.clip(S, 0.0, 1.0)
  return jnp.acos(S)


def get_qm_reconstructor(shifted_svd: ShiftedSVD, train_data):
  """
  Data reconstruction in quadratic manifold
  """
  sampling_basis = get_sampling_basis(shifted_svd)
  sampling_args = {
      'basis': sampling_basis,
      'full_dim': sampling_basis.shape[0],
      'reduced_dim': MANIDIM.value,
  }

  row_indices = sampler()(sampling_args)
  assert len(row_indices) == flags.FLAGS.n_sample_points
  logging.info('Collected samples %s', row_indices)
  train_data_row_idcs = train_data[row_indices, :] - shifted_svd.shift[
      row_indices, None]
  logging.info("Selecting column indices")
  idx_in = select_columns(shifted_svd, train_data_row_idcs, row_indices)
  assert len(idx_in) == MANIDIM.value

  leftsings, singvals, rightsingsT = (
      shifted_svd.U,
      shifted_svd.S,
      shifted_svd.VT,
  )
  singvals_in = singvals[idx_in]
  leftsings_in = leftsings[:, idx_in]
  rightsings_inT = rightsingsT[idx_in, :]

  basis = leftsings_in
  full_reduced_coordinates = singvals_in[:, None] * rightsings_inT
  reduced_coords_fn = REDUCTION_FUNCTIONS[REDUCTION_STYLE.value](
      basis,
      row_indices,
      train_data_row_idcs,
      full_reduced_coordinates,
      n_iterations=0,  # no newton here
  )
  reduced_train_points = reduced_coords_fn(train_data_row_idcs)
  logging.info(
      'Reduced coords. rel. err.: %s',
      jnp.linalg.norm(full_reduced_coordinates - reduced_train_points) /
      jnp.linalg.norm(full_reduced_coordinates))
  logging.info('Reduced train points shape: %s', reduced_train_points.shape)
  logging.info('Right singular vectors shape: %s', rightsingsT.shape)

  padded_basis = jnp.zeros(rightsingsT.shape)
  padded = padded_basis.at[idx_in, :].set(reduced_train_points)
  nonlinear_coeffT = jnp.linalg.lstsq(
      kron_map(reduced_train_points).T,
      -padded.T + (singvals[:, None] * rightsingsT).T,
      # -(basis @ reduced_train_points).T + train_data.T,
      rcond=RECONSTRUCTION_LSTSQ_REGMAG.value,
  )[0]
  nonlinear_coeff = leftsings @ nonlinear_coeffT.T
  deco = deim_constant(
      basis,
      row_indices,
  )
  exp.add_result(
      flags.FLAGS.hash,
      {
          'deim_constant': float(deco),
      },
  )

  reduced_coords_fn = REDUCTION_FUNCTIONS[REDUCTION_STYLE.value](
      basis,
      row_indices,
      train_data_row_idcs,
      full_reduced_coordinates,
      nonlinear_coeff=nonlinear_coeff,
  )

  def reconstruct(sampled_data_points):
    red_points = reduced_coords_fn(sampled_data_points)
    assert red_points.shape[0] == MANIDIM.value
    return basis @ red_points + nonlinear_coeff @ kron_map(red_points)

  return shift_wrapper(reconstruct, shifted_svd.shift, row_indices), row_indices


def upper_arrow(x):
  A = jnp.diag(x[-1] * jnp.ones(x.shape[0]))
  A = A.at[-1, -1].set(2 * A[-1, -1])
  A = A.at[:-1, -1].set(x[:-1])
  return A


def kron_map_deriv(x):
  r = x.shape[0]
  cc = []
  for i in range(r):
    cc.append(
        jnp.hstack((upper_arrow(x[:i + 1]), jnp.zeros((i + 1, r - i - 1)))))
  return jnp.vstack(cc)


def select_columns(shifted_svd: ShiftedSVD, train_data_row_idcs, row_indices):
  if GREEDY_COL_IDX.value == False:
    idx_in = jnp.arange(MANIDIM.value)
  else:
    idx_in = jnp.arange(0)
    # train_data_row_idcs = train_data[row_indices, :]
    leftsings = shifted_svd.U
    sigmaVT = shifted_svd.S[:, None] * shifted_svd.VT
    sigmaVTT = sigmaVT.T
    greedy_step_counter = 1
    while len(idx_in) < MANIDIM.value:
      idx_in = greedy_step(
          train_data_row_idcs,
          row_indices,
          idx_in,
          leftsings,
          sigmaVT,
          sigmaVTT,
          GREEDY_NCONSIDER.value,
          kron_map,
          RECONSTRUCTION_LSTSQ_REGMAG.value,
      )
      greedy_step_counter += 1

  return idx_in


def greedy_step(
    train_data_row_idcs,
    row_indices,
    idx_in,
    leftsings,
    sigmaVT,
    sigmaVTT,
    imax,
    nonlinear_map,
    reg_magnitude,
):

  padded_basis = jnp.zeros(sigmaVTT.shape)

  def compute(i, rec_errs):
    test_col_idcs = jnp.hstack((*idx_in, i))
    reduced_coords_fn = REDUCTION_FUNCTIONS[REDUCTION_STYLE.value](
        leftsings[:, test_col_idcs],
        row_indices,
        train_data_row_idcs,
        sigmaVT[test_col_idcs, :],
        n_iterations=0,
    )
    red_coords = reduced_coords_fn(train_data_row_idcs)
    padded = padded_basis.at[:, test_col_idcs].set(red_coords.T)
    _, resids, _, _ = jnp.linalg.lstsq(
        nonlinear_map(red_coords).T,
        (-padded + sigmaVTT),
        rcond=reg_magnitude,
    )
    rec_err = jnp.linalg.norm(resids)
    rec_errs = rec_errs.at[i].set(rec_err)
    return rec_errs

  def setinf(i, rec_errs):
    rec_errs = rec_errs.at[i].set(jnp.inf)
    return rec_errs

  def body_fun(i, rec_errs):
    return jax.lax.cond(
        jnp.min(jnp.abs(idx_in - i)) == 0,
        setinf,
        compute,
        i,
        rec_errs,
    )

  rec_errs = jnp.zeros(imax)
  if len(idx_in) == 0:
    rec_errs = jax.lax.fori_loop(0, imax, compute, rec_errs)
  else:
    rec_errs = jax.lax.fori_loop(0, imax, body_fun, rec_errs)
  idx_in = jnp.hstack((*idx_in, jnp.argmin(rec_errs)))
  return idx_in
