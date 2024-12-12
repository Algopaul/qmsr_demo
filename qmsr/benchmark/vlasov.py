"""Driver script for simulating collisionless particles"""
import jax.numpy as jnp
from absl import flags
from absl import app
from absl import logging
from qmsr.benchmark.integration import get_all_stepper
import h5py
import matplotlib.pyplot as plt
import jax

jax.config.update('jax_enable_x64', True)

T0 = flags.DEFINE_float('ode_t0', 0.0, 'ode start time')
T1 = flags.DEFINE_float('ode_t1', 1.0, 'ode end time')
DT = flags.DEFINE_float('ode_dt', 2e-3, 'ode time step')
PLOTTING = flags.DEFINE_boolean(
    'plotting',
    False,
    'Whether to plot the snapshots.',
)
STORING = flags.DEFINE_boolean(
    'storing',
    False,
    'Whether to store the trajectory',
)


def pulse2d(x):
  bump_shifts = jnp.asarray([-0.0, -0.0])
  sharpness = -100
  dist_x = jnp.square(x[0] - bump_shifts[0])
  dist_y = jnp.square(x[1] - bump_shifts[1])
  return jnp.exp(sharpness * (dist_x + dist_y))


def phi(x):
  return (0.2 + 0.2 * jnp.cos(jnp.pi * x**4) + 0.1 * jnp.sin(jnp.pi * x))


def get_vlasov_2d(d_dx, d_dy, x2, d_dxphi):

  def vlasov_2d(t, u, _):
    del t
    return -x2 * d_dx(u) - d_dxphi * d_dy(u)

  return vlasov_2d


def get_central_deriv(stepsize, axis):
  h2 = 2 * stepsize

  def deriv(field):
    diff = jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis)
    return diff / h2

  return deriv


def main(_):
  n_gridcells = 600
  logging.info("2D Wave trajectory: ODE setup")
  timepoints = jnp.arange(T0.value, T1.value, DT.value)
  train_times = timepoints[::2]
  test_times = timepoints[1::2]
  x = jnp.linspace(-1.0, 1.0, n_gridcells, endpoint=False)
  stepsize = x[1] - x[0]
  grid = jnp.meshgrid(x, x)
  phi_dx_data = get_central_deriv(stepsize, 0)((phi(grid[1])))
  ode = get_vlasov_2d(
      get_central_deriv(stepsize, 0),
      get_central_deriv(stepsize, 1),
      grid[0],
      phi_dx_data,
  )
  stepper = get_all_stepper(ode, timepoints)
  u0 = pulse2d(grid)
  logging.info("Vlasov trajectory: Start integration")
  trajectory = stepper(u0, None)
  logging.info("Vlasov trajectory: Finish integration")
  if PLOTTING.value:
    idcs = jnp.floor(jnp.linspace(0, len(trajectory) - 1, 10)).astype('int')
    for j, i in enumerate(idcs):
      fig = plt.figure()
      plt.imshow(trajectory[i])
      plt.savefig(f'vlasov_2d_{j}.png')
      plt.close(fig)

  if STORING.value:
    logging.info("Vlasov trajectory: Split trajectories results")
    trajectory = trajectory.reshape(len(timepoints), -1)
    train_data = trajectory[::2]
    test_data = trajectory[1::2]
    logging.info("Vlasov trajectory: Store as hdf5")
    with h5py.File("data/datafiles/vlasov.h5", "w") as f:
      f.create_dataset('train_data', data=train_data.T)
      f.create_dataset('train_times', data=train_times)
      f.create_dataset('test_data', data=test_data.T)
      f.create_dataset('test_times', data=test_times)


if __name__ == '__main__':
  app.run(main)
