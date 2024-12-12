from absl.testing import absltest
import jax.numpy as jnp
import jax
from qmsr.reconstruction.sampling import oversampled_qdeim_points

jax.config.update('jax_enable_x64', True)


class OversampledQDEIMTest(absltest.TestCase):

  def test_oversampled_qdeim(self):
    """
    Testing if oversampled qdeim routine from
    https://github.com/pehersto/odeim/blob/master/gpode.m
    is translated correctly.
    """
    A = jax.random.uniform(jax.random.key(0), (100, 20))
    U, _, _ = jnp.linalg.svd(A, full_matrices=False)

    # Target idcs computed with code at github.com/pehersto/odeim/
    target_idcs = jnp.sort(
        jnp.asarray([
            6, 84, 95, 60, 81, 70, 4, 21, 43, 77, 36, 92, 44, 76, 66, 51, 69,
            62, 79, 54, 32, 19, 97, 14, 83, 38, 49, 89, 52, 8
        ]))

    idcs = oversampled_qdeim_points(U, 30)
    self.assertSequenceEqual(list(target_idcs), list((idcs + 1).astype('int')))


if __name__ == "__main__":
  absltest.main()
