from exp import generate_table

group_field = 'manidim'
result_fields = ['test_err']


def greedy_constraints(encoder_iterations, gm_damping=None):
  cs = [
      "reconstruction_style='qmsr'",
      "greedy_col_idx=True",
      "reduction_style='deim_iter'",
      f"encoder_iterations={encoder_iterations}",
  ]
  if gm_damping is not None:
    cs.append(f'gauss_newton_damping={gm_damping}')
  return cs


columns = [
    {
        'name':
            'DEIM',
        'constraints': [
            "reconstruction_style='linear'",
            "reduction_style='deim'",
        ]
    },
    {
        'name': 'QMSRGreedy0',
        'constraints': greedy_constraints(0)
    },
    {
        'name': 'QMSRGreedyBestof',
        'constraints':
            greedy_constraints(20),  # picks damping with lowest error
    },
]

factors = [1, 2, 3, 4]
inflated = [True, False]
for factor in factors:
  for infl in inflated:
    extra_flags = [
        f'{factor}*manidim=n_sample_points',
        "data_file like '%vlasov%'",
    ]
    if infl:
      extra_flags.append('qdeim_inflated_basis=True')
    else:
      extra_flags.append('qdeim_inflated_basis=False')
    generate_table(
        'manidim',
        result_fields,
        extra_flags,
        columns,
        experiment_name='vlasov',
        tablenames=[
            f'vlasov_rx{factor}r_{infl}',
        ],
        secondary='gauss_newton_damping',
    )

columns = [
    {
        'name':
            'LinFull',
        'constraints': [
            "reconstruction_style='linear'",
            "greedy_col_idx=False",
            "reduction_style='full'",
        ]
    },
    {
        'name':
            'QMFull',
        'constraints': [
            "reconstruction_style='qmsr'",
            "greedy_col_idx=False",
            "reduction_style='full'",
        ]
    },
    {
        'name':
            'QMGreedyFull',
        'constraints': [
            "reconstruction_style='qmsr'",
            "greedy_col_idx=True",
            "reduction_style='full'",
        ]
    },
]
generate_table(
    'manidim',
    result_fields,
    ["tag='vlasov_rel_errs'"],
    columns,
    experiment_name='vlasov',
    tablenames=[
        f'vlasov_full',
    ],
)
