#!/bin/bash

trap "kill 0" EXIT

# active learning:
python train.py --config_filename='data/model/poisson_active.yaml'

# 3 cases for offline learning:
# # full
# python train.py --config_filename='data/model/poisson_full.yaml'

# # nested
# python train.py --config_filename='data/model/poisson_nested.yaml'

# # disjoint
# python train.py --config_filename='data/model/poisson_disjoint.yaml'


# wait