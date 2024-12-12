import exp
import os
from absl import app
from absl import flags

USE_SLURM = flags.DEFINE_boolean(
    'use_slurm',
    True,
    'Whether to use SLURM when running experiments',
)

with open('config/exp_run') as f:
  rcmd = f.readline()

if "srun" in rcmd:
  suf = '& '
else:
  suf = ''


def base_command(config_dict):
  """
    Add slurm logs to data/logs/{config_dict['hash']}{.log | .err}.
    Construct the base slurm command with log flags.
    
    Args:
        config_dict (dict): Dictionary containing configuration parameters, including 'hash'.
    
    Returns:
        str: The base command string with the appropriate log paths and command flags.
    """
  log_dir = os.path.join('data', 'logs')
  log_file = os.path.join(log_dir, f"{config_dict['hash']}.log")
  err_file = os.path.join(log_dir, f"{config_dict['hash']}.err")
  sflags = " --mem 32G --time 0:30:00 --cpus-per-task=4"

  # Define slurm log flags
  log_flags = f"--output={log_file} --error={err_file}"

  # Base command with log paths included
  if USE_SLURM.value is True:
    base_cmd = f"{rcmd[:-1]} {sflags} {log_flags} .venv/bin/python qmsr/driver.py"
  else:
    base_cmd = f".venv/bin/python qmsr/driver.py"

  return base_cmd


def main(_):
  with open('run.sh', 'w') as f:
    f.write(
        exp.generate_commands(
            base_command,
            'config_flags.cfg',
            'data/db.sqlite3',
            rerun_failed=True,
            rerun_scheduled=True,
            suffix=suf,
        ))


if __name__ == '__main__':
  app.run(main)
