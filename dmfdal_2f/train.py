from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from lib import utils
from model.pytorch.supervisor import Supervisor
import random
import numpy as np
import os
import pickle


def main(args):

    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

    max_itr = supervisor_config.get('train').get('max_itr', 25) #25
    seed = supervisor_config.get('train').get('seed', 1) #25
    costs = supervisor_config.get('train').get('costs')
    opt_rate = supervisor_config.get('train').get('opt_lr')
    acq_weight = supervisor_config.get('train').get('acq_weight')
    num_sample = supervisor_config.get('train').get('num_sample')
    data_type = supervisor_config.get('data').get('data_type')
    method = supervisor_config.get('train').get('method')
    fidelity_weight = supervisor_config.get('train').get('fidelity_weight')

    np.random.seed(seed)
    random.seed(seed)

    data = utils.load_dataset(**supervisor_config.get('data'))
    supervisor = Supervisor(random_seed=seed, iteration=0, max_itr = max_itr, **supervisor_config)

    # if not os.path.exists('seed%d/reward_list' % (i)): #for nRmse
    #         os.makedirs('seed%d/reward_list' % (i))
    if not os.path.exists('results'): #for cost
            os.makedirs('results')

    m_batch_list = []
    fidelity_info_list = []
    fidelity_query_list = []
    reg_info_list = []
    l2_y_preds_all = []
    
    

    test_nll_list = []
    test_rmse_list = []
    test_nrmse_list = []

    for itr in range(max_itr):
        supervisor._data = data
        supervisor.iteration = itr
        l1_x_s, l1_y_s, l2_x_s, l2_y_s, m_batch, fidelity_info, fidelity_query, reg_info, test_nll, test_rmse, test_nrmse, l2_y_truths, l2_y_preds_mu = supervisor.train()


        selected_data = {}
        selected_data['l1_x'] = l1_x_s
        selected_data['l1_y'] = l1_y_s
        selected_data['l2_x'] = l2_x_s
        selected_data['l2_y'] = l2_y_s
        search_config = supervisor_config.get('data').copy()
        search_config['selected_data'] = selected_data
        search_config['previous_data'] = data
        data = utils.generate_new_trainset(**search_config)

        m_batch_list.append(m_batch)
        fidelity_info_list.append(fidelity_info)
        fidelity_query_list.append(fidelity_query)
        reg_info_list.append(reg_info)

        test_nll_list.append(test_nll)
        test_rmse_list.append(test_rmse)
        test_nrmse_list.append(test_nrmse)

        # m_batch = np.stack(m_batch_list)
        # fidelity_info = np.stack(fidelity_info_list)
        # fidelity_query = np.stack(fidelity_query_list).squeeze()
        # reg_info = np.stack(reg_info_list)

        # test_nll = np.stack(test_nll_list)
        # test_rmse = np.stack(test_rmse_list)
        # test_nrmse = np.stack(test_nrmse_list)

        dictionary = {'fidelity': m_batch_list, 'score': fidelity_info_list, 'x': fidelity_query_list, 'weighted_score': reg_info_list, 'nll': test_nll_list, 'rmse': test_rmse_list, 'nrmse': test_nrmse_list}

        with open('results/exp_'+str(data_type)+'_opt_'+str(method)+'_fweight_'+str(fidelity_weight)+'_optlr'+str(opt_rate)+'_weight'+str(acq_weight)+'_sample'+str(num_sample)+'_cost'+str(costs[-1])+'_seed'+str(seed)+'.pkl', 'wb') as f:
            pickle.dump(dictionary, f)

        print('l2_y_truths.shape',l2_y_truths.shape)
        print('l2_y_preds_mu.shape',l2_y_preds_mu.shape)
        l2_y_preds_all.append(l2_y_preds_mu)
        print('l2_y_preds_all size: ', len(l2_y_preds_all))

    np.save('results/exp'+str(data_type)+'_opt'+str(method)+'_sample'+str(num_sample)+'_seed'+str(seed)+'truths.npz', l2_y_truths)
    np.save('results/exp'+str(data_type)+'_opt'+str(method)+'_sample'+str(num_sample)+'_seed'+str(seed)+'preds_mu.npz', l2_y_preds_all)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/seed1.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)



