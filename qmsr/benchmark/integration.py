"""Tools for integrating ODEs on regular time grid"""
import jax
import jax.numpy as jnp
from absl import logging


def rk4_step(x, t0, t1, rhs):
  delta_t = t1 - t0
  k1 = rhs(t0, x)
  k2 = rhs(t0 + delta_t / 2, x + delta_t * k1 / 2)
  k3 = rhs(t0 + delta_t / 2, x + delta_t * k2 / 2)
  k4 = rhs(t1, x + delta_t * k3)
  return x + 1.0 / 6.0 * delta_t * (k1 + 2 * k2 + 2 * k3 + k4)


def get_all_stepper(
    ode,
    timepoints,
    bc_callback=lambda x: x,
):
  logging.info('Generating time stepper that stores all time steps')
  dt = timepoints[1] - timepoints[0]

  def steps(x0, params):

    def rhs(t, x):
      return ode(t, x, params)

    def inner_step(carry, t):
      xp = rk4_step(carry, t, t + dt, rhs)
      xp = bc_callback(xp)
      return xp, xp

    _, time_series = jax.lax.scan(inner_step, x0, timepoints)
    return time_series

  return steps


def get_stepper(
    ode,
    timepoints,
    supersample_factor,
    bc_callback=lambda x: x,
):
  logging.info('Generating time stepper that stores every %d-th value',
               supersample_factor)
  t = supsamplinspace(timepoints, supersample_factor)
  dt = t[1] - t[0]

  def steps(x0, params):

    def rhs(t, x):
      return ode(t, x, params)

    def fori_step(i, carry):
      del i
      x, t = carry
      xp = rk4_step(x, t, t + dt, rhs)
      xp = bc_callback(xp)
      return (xp, t + dt)

    def scan_step(xp, t):
      xp, _ = jax.lax.fori_loop(0, supersample_factor, fori_step, (xp, t))
      return xp, xp

    t_sub = t[::supersample_factor]
    _, time_series = jax.lax.scan(scan_step, x0, t_sub)
    return time_series

  return steps


def supsamplinspace(t, supersample_factor=1):
  num_samples = len(t) + (len(t) - 1) * (supersample_factor - 1)
  return jnp.linspace(t[0], t[-1], num_samples)
