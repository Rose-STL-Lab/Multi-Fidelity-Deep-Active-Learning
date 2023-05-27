#!/bin/bash

trap "kill 0" EXIT

# active learning:
python train.py --config_filename='data/model/heat_active.yaml'

# 3 cases for offline learning:
# # full
# python train.py --config_filename='data/model/heat_full.yaml'

# # nested
# python train.py --config_filename='data/model/heat_nested.yaml'

# # disjoint
# python train.py --config_filename='data/model/heat_disjoint.yaml'


# wait