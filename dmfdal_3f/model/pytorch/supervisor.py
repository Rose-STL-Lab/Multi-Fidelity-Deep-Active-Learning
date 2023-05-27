import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.model import Model
from model.pytorch.loss import nll_loss
from model.pytorch.loss import nll_metric
from model.pytorch.loss import rmse_metric
# from model.pytorch.loss import nonormalized_mae_metric
from model.pytorch.loss import kld_gaussian_loss
from torch.utils.tensorboard import SummaryWriter
import model.pytorch.dataset_active as dataset
from torch.optim import LBFGS
import torch.nn as nn
import copy
import sys

import csv


# device = torch.device("cuda:1")

class Supervisor:
    def __init__(self, random_seed, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._data_type = self._data_kwargs.get('data_type')
        self.synD = dataset.Dataset(self._data_type, random_seed)
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.costs = self._train_kwargs.get('costs', [1,3])
        self.acq_weight = self._train_kwargs.get('acq_weight', 1e-2)
        self.method = self._train_kwargs.get('method')
        self.num_sample = int(self._train_kwargs.get('num_sample', 100))
        self.fidelity_weight = self._train_kwargs.get('fidelity_weight', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs, self.random_seed, self.costs, self.acq_weight, self.num_sample, self._data_type, self.method, self.fidelity_weight)
        # self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.x_scaler = self._data['l1_x_scaler']
        self.l1_y_scaler = self._data['l1_y_scaler']
        self.l2_y_scaler = self._data['l2_y_scaler']
        self.l3_y_scaler = self._data['l3_y_scaler']

        self.input_dim = int(self._model_kwargs.get('input_dim', 3))
        self.l1_output_dim = int(self._model_kwargs.get('l1_output_dim', 256))
        self.l2_output_dim = int(self._model_kwargs.get('l2_output_dim', 1024))
        self.l3_output_dim = int(self._model_kwargs.get('l3_output_dim', 4096))
        self.z_dim = int(self._model_kwargs.get('z_dim',32))
        self.num_batches = None #int(0)
        self.device_num = self._model_kwargs.get('device') #"cuda:5"

        self.device = torch.device(self.device_num) 
        self.budget = int(self._train_kwargs.get('budget', 20))
        self.opt_lr = self._train_kwargs.get('opt_lr', 1e-4)
        # self.opt_iter = self._train_kwargs.get('opt_iter', 2000)
        self.opt_every_n_epochs = self._train_kwargs.get('opt_every_n_epochs', 1)


        # setup model
        model = Model(self._logger, **self._model_kwargs)
        self.model = model.cuda(self.device) if torch.cuda.is_available() else model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        self._opt_epoch_num = self._train_kwargs.get('opt_epoch', 0)
        self._opt_epochs = self._train_kwargs.get('opt_epochs', 1000)

    @staticmethod
    def _get_log_dir(kwargs, random_seed, costs, acq_weight, num_sample, data_type, method, fidelity_weight):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            learning_rate = kwargs['train'].get('base_lr')
            opt_rate = kwargs['train'].get('opt_lr')
            # max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            # num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            # rnn_units = kwargs['model'].get('rnn_units')
            # structure = '-'.join(
                # ['%d' % rnn_units for _ in range(num_rnn_layers)])
            # horizon = kwargs['model'].get('horizon')

            run_id = 'exp_%s_opt_%s_fweight_%g_optlr_%g_lr_%g_weight_%g_sample_%d_cost_%d_seed_%d_%s/' % (
                data_type, method, fidelity_weight, opt_rate, learning_rate,
                acq_weight, num_sample, costs[-1], random_seed, time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)

        l1_z_mu_all, l1_z_cov_all, l2_z_mu_all, l2_z_cov_all, l3_z_mu_all, l3_z_cov_all, l1_test_nll, l1_test_rmse, l1_test_nrmse, l2_test_nll, l2_test_rmse, l2_test_nrmse, l3_test_nll, l3_test_rmse, l3_test_nrmse, l3_y_truths_scaled, l3_y_preds_mu_scaled = self._train(**kwargs)
        self._logger.info('l1_z_mu_all: {}, l1_z_cov_all: {}, l2_z_mu_all: {}, l2_z_cov_all: {}, l3_z_mu_all: {}, l3_z_cov_all: {}'.format(l1_z_mu_all, l1_z_cov_all, l2_z_mu_all, l2_z_cov_all, l3_z_mu_all, l3_z_cov_all))
        if self.method == "gradient":
            l1_x_s, l2_x_s, l3_x_s, m_batch, fidelity_info, fidelity_query, reg_info = self.submod_batch_query(l1_z_mu_all, l1_z_cov_all, l2_z_mu_all, l2_z_cov_all, l3_z_mu_all, l3_z_cov_all, self.budget)
        elif self.method == "random":
            l1_x_s, l2_x_s, l3_x_s, m_batch, fidelity_info, fidelity_query, reg_info = self.random_batch_query(l1_z_mu_all, l1_z_cov_all, l2_z_mu_all, l2_z_cov_all, l3_z_mu_all, l3_z_cov_all, self.budget)

        mu = self.x_scaler.mean
        std = self.x_scaler.std

        fidelity_query = fidelity_query * std + mu

        self._logger.info('score_info: {}'.format(fidelity_info))
        self._logger.info('weighted_score_info: {}'.format(reg_info))
        if len(l1_x_s) != 0:
            l1_y_s = self.synD.multi_query(l1_x_s, 0, mu, std)
        else:
            l1_y_s = np.empty((0, self.l1_output_dim))

        # l2_data_size = self._data['l2_train_loader'].size

        if len(l2_x_s) != 0:
            l2_y_s = self.synD.multi_query(l2_x_s, 1, mu, std)
        else:
            l2_y_s = np.empty((0, self.l2_output_dim))

        if len(l3_x_s) != 0:
            l3_y_s = self.synD.multi_query(l3_x_s, 2, mu, std)
        else:
            l3_y_s = np.empty((0, self.l3_output_dim))

        test_nll = np.array([l1_test_nll, l2_test_nll, l3_test_nll])
        test_rmse = np.array([l1_test_rmse, l2_test_rmse , l3_test_rmse])
        test_nrmse = np.array([l1_test_nrmse, l2_test_nrmse, l3_test_nrmse])

        return l1_x_s, l1_y_s, l2_x_s, l2_y_s, l3_x_s, l3_y_s, m_batch, fidelity_info, fidelity_query, reg_info, test_nll, test_rmse, test_nrmse, l3_y_truths_scaled, l3_y_preds_mu_scaled

    def init_query_points(self, m, Nq=1):
        lb, ub = self.synD.get_N_bounds()
        mean = self.x_scaler.mean
        std = self.x_scaler.std
        lb = (lb - mean)/std
        ub = (ub - mean)/std
        scale = (ub-lb).reshape([1,-1])
        uni_noise = np.random.uniform(size=[Nq, self.input_dim])
        
        np_Xq_init = uni_noise*scale + lb
        
        Xq = torch.tensor(np_Xq_init, requires_grad=True, dtype=torch.float32, device=self.device_num)
        
        return Xq, lb, ub

    def opt_submod_query(self, l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov, m):

        Xq, lb, ub = self.init_query_points(m)

        bounds = torch.tensor(np.vstack((lb, ub))).to(self.device)

        # lbfgs = LBFGS([Xq], lr=self.opt_lr, max_iter=self.opt_iter, max_eval=None)
        lbfgs = LBFGS([Xq], lr=self.opt_lr, max_eval=None)
        # lbfgs = torch.optim.Adam([Xq], lr=self.opt_lr)

        new_model = copy.deepcopy(self.model)
        new_model.train()

        l1_z_mu = l1_z_mu.detach()
        l1_z_cov = l1_z_cov.detach()
        l2_z_mu = l2_z_mu.detach()
        l2_z_cov = l2_z_cov.detach()
        l3_z_mu = l3_z_mu.detach()
        l3_z_cov = l3_z_cov.detach()

        def closure():
            lbfgs.zero_grad()
            # self._logger.info('m{}'.format(m))
            if m == 0:
                # zs = self.model.sample_z(l1_z_mu, l1_z_cov, Xq.size(0)).detach()
                zs = torch.repeat_interleave(l1_z_mu.unsqueeze(0), Xq.size(0), 0)

            elif m == 1:
                # zs = self.model.sample_z(l2_z_mu, l2_z_cov, Xq.size(0)).detach()
                zs = torch.repeat_interleave(l2_z_mu.unsqueeze(0), Xq.size(0), 0)

            elif m == 2:
                # zs = self.model.sample_z(l2_z_mu, l2_z_cov, Xq.size(0)).detach()
                zs = torch.repeat_interleave(l3_z_mu.unsqueeze(0), Xq.size(0), 0)

            # self._logger.info('zs.is_leaf{}'.format(zs.is_leaf))

            Yq, _ = new_model.z_to_y(Xq, zs, level=m+1)
            if m == 0:
                r_mu, r_cov = new_model.xy_to_r_global(Xq, Yq, level=m+1)
                # r_cov = r_cov * self.acq_weight # weighted representation
            elif m == 1:
                r_mu, r_cov = new_model.xy_to_r_global(Xq, Yq, level=m+1)
            elif m == 2:
                r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
                r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
                r_mu = torch.cat([r_mu_global, r_mu_local],0)
                r_cov = torch.cat([r_cov_global, r_cov_local],0)
                # r_cov = r_cov * self.acq_weight # weighted representation

            l3_v = r_mu - l3_z_mu
            l3_w_cov_inv = 1 / r_cov
            l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
            l3_z_cov_new = l3_z_cov_new.clamp(min=1e-3, max=1.)
            l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)
            l3_z_mu_new = l3_z_mu_new.clamp(min=-3.5, max=3.5)

            gain = kld_gaussian_loss(l3_z_mu_new, l3_z_cov_new, l3_z_mu, l3_z_cov)

            loss = -gain
            # loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)

            with torch.no_grad():
                for j, (lb, ub) in enumerate(zip(*bounds)):
                    Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself

            torch.nn.utils.clip_grad_norm_([Xq], self.max_grad_norm)

            return loss

        for epoch_num in range(self._opt_epoch_num, self._opt_epochs):
            # loss = closure()
            loss = lbfgs.step(closure)
            # print('Xq: ', Xq)
            # print('Xq.grad: ', Xq.grad)
            # sys.exit()

            log_every = self.opt_every_n_epochs
            if (epoch_num % log_every) == log_every - 1:
                message = 'Gradient optimization Epoch [{}/{}] ' \
                          'opt_loss: {:.4f}'.format(epoch_num, self._opt_epochs,
                                           loss)
                self._logger.info(message)

        # loss = lbfgs.step(closure)
        gain = -loss
        # print('Xq: ', Xq)

        #update z
        new_model.eval()
        with torch.no_grad():

            if m == 0:
                zs = torch.repeat_interleave(l1_z_mu.unsqueeze(0), Xq.size(0), 0)
                Yq, _ = new_model.z_to_y(Xq, zs[0:1], level=m+1)
                r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
                r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
                r_mu = torch.cat([r_mu_global, r_mu_local],0)
                r_cov = torch.cat([r_cov_global, r_cov_local],0)

                l1_v = r_mu - l1_z_mu
                l1_w_cov_inv = 1 / r_cov
                l1_z_cov_new = 1 / (1 / l1_z_cov + torch.sum(l1_w_cov_inv, dim=0))
                l1_z_mu_new = l1_z_mu + l1_z_cov_new * torch.sum(l1_w_cov_inv * l1_v, dim=0)

                l2_v = r_mu_global - l2_z_mu
                l2_w_cov_inv = 1 / r_cov_global
                l2_z_cov_new = 1 / (1 / l2_z_cov + torch.sum(l2_w_cov_inv, dim=0))
                l2_z_mu_new = l2_z_mu + l2_z_cov_new * torch.sum(l2_w_cov_inv * l2_v, dim=0)

                l3_v = r_mu_global - l3_z_mu
                l3_w_cov_inv = 1 / r_cov_global
                l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
                l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)

            elif m == 1:
                zs = torch.repeat_interleave(l2_z_mu.unsqueeze(0), Xq.size(0), 0)
                Yq, _ = new_model.z_to_y(Xq, zs[0:1], level=m+1)
                r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
                r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
                r_mu = torch.cat([r_mu_global, r_mu_local],0)
                r_cov = torch.cat([r_cov_global, r_cov_local],0)

                l1_v = r_mu_global - l1_z_mu
                l1_w_cov_inv = 1 / r_cov_global
                l1_z_cov_new = 1 / (1 / l1_z_cov + torch.sum(l1_w_cov_inv, dim=0))
                l1_z_mu_new = l1_z_mu + l1_z_cov_new * torch.sum(l1_w_cov_inv * l1_v, dim=0)

                l2_v = r_mu - l2_z_mu
                l2_w_cov_inv = 1 / r_cov
                l2_z_cov_new = 1 / (1 / l2_z_cov + torch.sum(l2_w_cov_inv, dim=0))
                l2_z_mu_new = l2_z_mu + l2_z_cov_new * torch.sum(l2_w_cov_inv * l2_v, dim=0)

                l3_v = r_mu_global - l3_z_mu
                l3_w_cov_inv = 1 / r_cov_global
                l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
                l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)

            elif m == 2:
                zs = torch.repeat_interleave(l3_z_mu.unsqueeze(0), Xq.size(0), 0)
                Yq, _ = new_model.z_to_y(Xq, zs[0:1], level=m+1)
                r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
                r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
                r_mu = torch.cat([r_mu_global, r_mu_local],0)
                r_cov = torch.cat([r_cov_global, r_cov_local],0)

                l1_v = r_mu_global - l1_z_mu
                l1_w_cov_inv = 1 / r_cov_global
                l1_z_cov_new = 1 / (1 / l1_z_cov + torch.sum(l1_w_cov_inv, dim=0))
                l1_z_mu_new = l1_z_mu + l1_z_cov_new * torch.sum(l1_w_cov_inv * l1_v, dim=0)

                l2_v = r_mu_global - l2_z_mu
                l2_w_cov_inv = 1 / r_cov_global
                l2_z_cov_new = 1 / (1 / l2_z_cov + torch.sum(l2_w_cov_inv, dim=0))
                l2_z_mu_new = l2_z_mu + l2_z_cov_new * torch.sum(l2_w_cov_inv * l2_v, dim=0)

                l3_v = r_mu - l3_z_mu
                l3_w_cov_inv = 1 / r_cov
                l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
                l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)

        return gain, Xq, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new

    def random_query(self, l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov, m):

        Xq, lb, ub = self.init_query_points(m, self.num_sample)

        new_model = copy.deepcopy(self.model)

        l1_z_mu = l1_z_mu.detach()
        l1_z_cov = l1_z_cov.detach()
        l2_z_mu = l2_z_mu.detach()
        l2_z_cov = l2_z_cov.detach()
        l3_z_mu = l3_z_mu.detach()
        l3_z_cov = l3_z_cov.detach()

        if m == 0:
            # zs = self.model.sample_z(l1_z_mu, l1_z_cov, Xq.size(0)).detach()
            zs = torch.repeat_interleave(l1_z_mu.unsqueeze(0), Xq.size(0), 0)

        elif m == 1:
            # zs = self.model.sample_z(l2_z_mu, l2_z_cov, Xq.size(0)).detach()
            zs = torch.repeat_interleave(l2_z_mu.unsqueeze(0), Xq.size(0), 0)

        elif m == 2:
            zs = torch.repeat_interleave(l3_z_mu.unsqueeze(0), Xq.size(0), 0)

        Yq, _ = new_model.z_to_y(Xq, zs, level=m+1)
        if m == 0:
            r_mu, r_cov = new_model.xy_to_r_global(Xq, Yq, level=m+1)
            r_mu = r_mu.unsqueeze(1)
            r_cov = r_cov.unsqueeze(1)
            # print('r_mu shape: ', r_mu.shape)
        elif m == 1:
            r_mu, r_cov = new_model.xy_to_r_global(Xq, Yq, level=m+1)
            r_mu = r_mu.unsqueeze(1)
            r_cov = r_cov.unsqueeze(1)
        elif m == 2:
            r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
            r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
            r_mu = torch.stack([r_mu_global, r_mu_local],1)
            r_cov = torch.stack([r_cov_global, r_cov_local],1)
            # print('r_mu shape: ', r_mu.shape)

        # self._logger.info('r_cov shape: '+str(r_cov.shape))

        gain_list = []

        for i in range(len(r_mu)):

            l3_v = r_mu[i] - l3_z_mu
            l3_w_cov_inv = 1 / r_cov[i]
            l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
            l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)

            gain = kld_gaussian_loss(l3_z_mu_new, l3_z_cov_new, l3_z_mu, l3_z_cov).item()
            gain_list.append(gain)

        gain_list = np.array(gain_list)
        gain_min = np.min(gain_list)
        gain_max = np.max(gain_list)
        ind = np.argmax(gain_list)
        gain = gain_list[ind]

        #update z
        Xq = Xq[ind].unsqueeze(0)
        Yq, _ = new_model.z_to_y(Xq, zs[0:1], level=m+1)

        if m == 0:
            r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
            r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
            r_mu = torch.cat([r_mu_global, r_mu_local],0)
            r_cov = torch.cat([r_cov_global, r_cov_local],0)

            l1_v = r_mu - l1_z_mu
            l1_w_cov_inv = 1 / r_cov
            l1_z_cov_new = 1 / (1 / l1_z_cov + torch.sum(l1_w_cov_inv, dim=0))
            l1_z_mu_new = l1_z_mu + l1_z_cov_new * torch.sum(l1_w_cov_inv * l1_v, dim=0)

            l2_v = r_mu_global - l2_z_mu
            l2_w_cov_inv = 1 / r_cov_global
            l2_z_cov_new = 1 / (1 / l2_z_cov + torch.sum(l2_w_cov_inv, dim=0))
            l2_z_mu_new = l2_z_mu + l2_z_cov_new * torch.sum(l2_w_cov_inv * l2_v, dim=0)

            l3_v = r_mu_global - l3_z_mu
            l3_w_cov_inv = 1 / r_cov_global
            l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
            l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)

        elif m == 1:
            r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
            r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
            r_mu = torch.cat([r_mu_global, r_mu_local],0)
            r_cov = torch.cat([r_cov_global, r_cov_local],0)

            l1_v = r_mu_global - l1_z_mu
            l1_w_cov_inv = 1 / r_cov_global
            l1_z_cov_new = 1 / (1 / l1_z_cov + torch.sum(l1_w_cov_inv, dim=0))
            l1_z_mu_new = l1_z_mu + l1_z_cov_new * torch.sum(l1_w_cov_inv * l1_v, dim=0)

            l2_v = r_mu - l2_z_mu
            l2_w_cov_inv = 1 / r_cov
            l2_z_cov_new = 1 / (1 / l2_z_cov + torch.sum(l2_w_cov_inv, dim=0))
            l2_z_mu_new = l2_z_mu + l2_z_cov_new * torch.sum(l2_w_cov_inv * l2_v, dim=0)

            l3_v = r_mu_global - l3_z_mu
            l3_w_cov_inv = 1 / r_cov_global
            l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
            l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)

        elif m == 2:
            r_mu_global, r_cov_global = new_model.xy_to_r_global(Xq, Yq, level=m+1)
            r_mu_local, r_cov_local = new_model.xy_to_r_local(Xq, Yq, level=m+1)
            r_mu = torch.cat([r_mu_global, r_mu_local],0)
            r_cov = torch.cat([r_cov_global, r_cov_local],0)

            l1_v = r_mu_global - l1_z_mu
            l1_w_cov_inv = 1 / r_cov_global
            l1_z_cov_new = 1 / (1 / l1_z_cov + torch.sum(l1_w_cov_inv, dim=0))
            l1_z_mu_new = l1_z_mu + l1_z_cov_new * torch.sum(l1_w_cov_inv * l1_v, dim=0)

            l2_v = r_mu - l2_z_mu
            l2_w_cov_inv = 1 / r_cov
            l2_z_cov_new = 1 / (1 / l2_z_cov + torch.sum(l2_w_cov_inv, dim=0))
            l2_z_mu_new = l2_z_mu + l2_z_cov_new * torch.sum(l2_w_cov_inv * l2_v, dim=0)

            l3_v = r_mu - l3_z_mu
            l3_w_cov_inv = 1 / r_cov
            l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
            l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)


        return gain, Xq, gain_min, gain_max, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new

    def submod_eval_next(self, l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov):

        fidelity_info = []
        fidelity_query = []
        fidelity_costs = []
        costs = self.costs #change [1,3], [1, 1]

        for m in range(3): #self.M
            info, xq, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new = self.opt_submod_query(l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov, m)
            fidelity_info.append(info.data.cpu().numpy())
            fidelity_query.append(xq)
            fidelity_costs.append(costs[m])
        #

        fidelity_info = np.array(fidelity_info)
        fidelity_costs = np.array(fidelity_costs)
        reg_info = fidelity_info / fidelity_costs
        
        argm = np.argmax(reg_info)
        argx = fidelity_query[argm]

        fidelity_query = torch.stack(fidelity_query).detach().cpu().numpy()

        self._logger.info('argm = '+str(argm))
        self._logger.info('argx = '+str(argx.data.cpu().numpy()))
        
        return argx, argm, fidelity_info, fidelity_query, reg_info, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new

    def random_eval_next(self, l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov):

        fidelity_info = []
        fidelity_query = []
        fidelity_costs = []
        costs = self.costs #change [1,3], [1, 1]

        for m in range(3): #self.M
            info, xq, gain_min, gain_max, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new = self.random_query(l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov, m)
            self._logger.info('fidelity '+ str(m) + ' min gain: '+str(gain_min))
            self._logger.info('fidelity '+ str(m) + ' max gain: '+str(gain_max))
            fidelity_info.append(info)
            fidelity_query.append(xq)
            fidelity_costs.append(costs[m])
        #

        fidelity_info = np.array(fidelity_info)
        fidelity_costs = np.array(fidelity_costs)
        reg_info = fidelity_info / fidelity_costs
        
        argm = np.argmax(reg_info)
        argx = fidelity_query[argm]

        fidelity_query = torch.stack(fidelity_query).detach().cpu().numpy()

        self._logger.info('argm = '+str(argm))
        self._logger.info('argx = '+str(argx.data.cpu().numpy()))
        
        return argx, argm, fidelity_info, fidelity_query, reg_info, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new

    def submod_batch_query(self, l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov, budget):

        B = budget
        query_costs = 0
        
        X_batch_l1 = []
        X_batch_l2 = []
        X_batch_l3 = []
        m_batch = []
        fidelity_info_list = []
        fidelity_query_list = []
        reg_info_list = []
        costs = self.costs

        while query_costs < B:
            argX, argm, fidelity_info, fidelity_query, reg_info, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new = self.submod_eval_next(l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov)
            m_batch.append(argm)
            if argm == 0:
                X_batch_l1.append(argX)
            elif argm == 1:
                X_batch_l2.append(argX)
            elif argm == 2:
                X_batch_l3.append(argX)

            fidelity_info_list.append(fidelity_info)
            fidelity_query_list.append(fidelity_query)
            reg_info_list.append(reg_info)


            # self._logger.info('m_batch: {}'.format(m_batch))
            current_costs = np.array([costs[m] for m in m_batch]).sum()
            # self._logger.info('current_costs: {}'.format(current_costs))
            query_costs = current_costs
            # update l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov
            l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov = l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new

        m_batch = np.stack(m_batch,0)

        if len(X_batch_l1) == 0:
            l1_x_s = np.empty((0, self.input_dim))
        else:
            l1_x_s = torch.cat(X_batch_l1,0).detach().cpu().numpy()
        if len(X_batch_l2) == 0:
            l2_x_s = np.empty((0, self.input_dim))
        else:
            l2_x_s = torch.cat(X_batch_l2,0).detach().cpu().numpy()
        if len(X_batch_l3) == 0:
            l3_x_s = np.empty((0, self.input_dim))
        else:
            l3_x_s = torch.cat(X_batch_l3,0).detach().cpu().numpy()
        
        
        self._logger.info('l1_x_s shape: {}, l2_x_s shape: {}, l3_x_s shape: {}, m_batch: {}'.format(l1_x_s.shape, l2_x_s.shape, l3_x_s.shape, m_batch))

        m_batch = np.stack(m_batch)
        fidelity_info = np.stack(fidelity_info_list)
        fidelity_query = np.stack(fidelity_query_list)
        reg_info = np.stack(reg_info_list)

        return l1_x_s, l2_x_s, l3_x_s, m_batch, fidelity_info, fidelity_query, reg_info

    def random_batch_query(self, l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov, budget):

        B = budget
        query_costs = 0
        
        X_batch_l1 = []
        X_batch_l2 = []
        X_batch_l3 = []
        m_batch = []
        fidelity_info_list = []
        fidelity_query_list = []
        reg_info_list = []
        costs = self.costs

        while query_costs < B:
            argX, argm, fidelity_info, fidelity_query, reg_info, l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new = self.random_eval_next(l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov)
            m_batch.append(argm)
            if argm == 0:
                X_batch_l1.append(argX)
            elif argm == 1:
                X_batch_l2.append(argX)
            elif argm == 2:
                X_batch_l3.append(argX)

            fidelity_info_list.append(fidelity_info)
            fidelity_query_list.append(fidelity_query)
            reg_info_list.append(reg_info)


            # self._logger.info('m_batch{}'.format(m_batch))
            current_costs = np.array([costs[m] for m in m_batch]).sum()
            # self._logger.info('current_costs{}'.format(current_costs))
            query_costs = current_costs
            # update l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov
            l1_z_mu, l1_z_cov, l2_z_mu, l2_z_cov, l3_z_mu, l3_z_cov = l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new


        m_batch = np.stack(m_batch,0)

        if len(X_batch_l1) == 0:
            l1_x_s = np.empty((0, self.input_dim))
        else:
            l1_x_s = torch.cat(X_batch_l1,0).detach().cpu().numpy()
        if len(X_batch_l2) == 0:
            l2_x_s = np.empty((0, self.input_dim))
        else:
            l2_x_s = torch.cat(X_batch_l2,0).detach().cpu().numpy()
        if len(X_batch_l3) == 0:
            l3_x_s = np.empty((0, self.input_dim))
        else:
            l3_x_s = torch.cat(X_batch_l3,0).detach().cpu().numpy()
        
        
        self._logger.info('l1_x_s shape: {}, l2_x_s shape: {}, l3_x_s shape: {}, m_batch: {}'.format(l1_x_s.shape, l2_x_s.shape, l3_x_s.shape, m_batch))

        m_batch = np.stack(m_batch)
        fidelity_info = np.stack(fidelity_info_list)
        fidelity_query = np.stack(fidelity_query_list)
        reg_info = np.stack(reg_info_list)

        return l1_x_s, l2_x_s, l3_x_s, m_batch, fidelity_info, fidelity_query, reg_info



    def evaluate(self, dataset='val', l1_z_mu_all=None, l1_z_cov_all=None, l2_z_mu_all=None, l2_z_cov_all=None, l3_z_mu_all=None, l3_z_cov_all=None):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model = self.model.eval()

            #change val_iterator
            # l1_test_iterator = self._data['l1_{}_loader'.format(dataset)].get_iterator()
            # l2_test_iterator = self._data['l2_{}_loader'.format(dataset)].get_iterator()

            l1_x_test = self._data['l1_{}_loader'.format(dataset)].xs
            l1_y_test = self._data['l1_{}_loader'.format(dataset)].ys
            l2_x_test = self._data['l2_{}_loader'.format(dataset)].xs
            l2_y_test = self._data['l2_{}_loader'.format(dataset)].ys
            l3_x_test = self._data['l3_{}_loader'.format(dataset)].xs
            l3_y_test = self._data['l3_{}_loader'.format(dataset)].ys

            l1_y_truths = []
            l2_y_truths = []
            l3_y_truths = []
            l1_y_preds_mu = []
            l2_y_preds_mu = []
            l3_y_preds_mu = []
            l1_y_preds_cov = []
            l2_y_preds_cov = []
            l3_y_preds_cov = []

            # for _, ((l1_x_test, l1_y_test), (l2_x_test, l2_y_test)) in enumerate(zip(l1_test_iterator, l2_test_iterator)): # need to be fixed
                # optimizer.zero_grad()

            x1_test, y1_test = self._test_l1_prepare_data(l1_x_test, l1_y_test) #train
            x2_test, y2_test = self._test_l2_prepare_data(l2_x_test, l2_y_test) #train
            x3_test, y3_test = self._test_l3_prepare_data(l3_x_test, l3_y_test) #train

            l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l3_output_mu, l3_output_cov = self.model(test=True, l1_x_test=x1_test, l2_x_test=x2_test, l3_x_test=x3_test, l1_z_mu_all=l1_z_mu_all, l1_z_cov_all=l1_z_cov_all, l2_z_mu_all=l2_z_mu_all, l2_z_cov_all=l2_z_cov_all, l3_z_mu_all=l3_z_mu_all, l3_z_cov_all=l3_z_cov_all)
            l1_y_truths.append(y1_test.cpu())
            l2_y_truths.append(y2_test.cpu())
            l3_y_truths.append(y3_test.cpu())
            l1_y_preds_mu.append(l1_output_mu.cpu())
            l2_y_preds_mu.append(l2_output_mu.cpu())
            l3_y_preds_mu.append(l3_output_mu.cpu())
            l1_y_preds_cov.append(l1_output_cov.cpu())
            l2_y_preds_cov.append(l2_output_cov.cpu())
            l3_y_preds_cov.append(l3_output_cov.cpu())

            l1_y_preds_mu = np.concatenate(l1_y_preds_mu, axis=0)
            l2_y_preds_mu = np.concatenate(l2_y_preds_mu, axis=0)
            l3_y_preds_mu = np.concatenate(l3_y_preds_mu, axis=0)

            l1_y_preds_cov = np.concatenate(l1_y_preds_cov, axis=0)
            l2_y_preds_cov = np.concatenate(l2_y_preds_cov, axis=0)
            l3_y_preds_cov = np.concatenate(l3_y_preds_cov, axis=0)

            l1_y_truths = np.concatenate(l1_y_truths, axis=0)
            l2_y_truths = np.concatenate(l2_y_truths, axis=0)
            l3_y_truths = np.concatenate(l3_y_truths, axis=0)

            l1_nll, l1_rmse, l1_nrmse, l2_nll, l2_rmse, l2_nrmse, l3_nll, l3_rmse, l3_nrmse, l3_y_truths_scaled, l3_y_preds_mu_scaled = self._test_loss(l1_y_preds_mu, l1_y_preds_cov, l1_y_truths, l2_y_preds_mu, l2_y_preds_cov, l2_y_truths, l3_y_preds_mu, l3_y_preds_cov, l3_y_truths)

            return l1_nll, l1_rmse, l1_nrmse, l2_nll, l2_rmse, l2_nrmse, l3_nll, l3_rmse, l3_nrmse, l3_y_truths_scaled, l3_y_preds_mu_scaled
            # , {'pred_mu': y_preds_mu, 'truth': y_truths}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        for epoch_num in range(self._epoch_num, epochs):
            # reshuffle the data
            self._data = utils.load_dataset(**self._data_kwargs)

            self.model = self.model.train()

            l1_x = self._data['l1_train_loader'].xs
            l1_y = self._data['l1_train_loader'].ys
            l2_x = self._data['l2_train_loader'].xs
            l2_y = self._data['l2_train_loader'].ys
            l3_x = self._data['l3_train_loader'].xs
            l3_y = self._data['l3_train_loader'].ys
            x_ref = self._data['x_ref']
            l1_y_ref = self._data['l1_y_ref']
            l2_y_ref = self._data['l2_y_ref']
            l3_y_ref = self._data['l3_y_ref']

            losses = []
            l1_nll_losses = []
            l2_nll_losses = []
            l3_nll_losses = []
            l1_kld_losses = []
            l2_kld_losses = []
            l3_kld_losses = []
            global_dist_losses = []


            start_time = time.time()

            x_ref, l1_y_ref, l2_y_ref, l3_y_ref = self._ref_prepare_data(x_ref, l1_y_ref, l2_y_ref, l3_y_ref)


            # for index, ((l1_x, l1_y), (l2_x, l2_y)) in enumerate(zip(l1_train_iterator, l2_train_iterator)): # need to be fixed
            optimizer.zero_grad()

            l1_x, l1_y = self._train_l1_prepare_data(l1_x, l1_y)
            l2_x, l2_y = self._train_l2_prepare_data(l2_x, l2_y)
            l3_x, l3_y = self._train_l3_prepare_data(l3_x, l3_y)


            l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l3_output_mu, l3_output_cov, l1_truth, l2_truth, l3_truth, l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c, l3_z_mu_all, l3_z_cov_all, l3_z_mu_c, l3_z_cov_c, l1_r_mu_ref, l1_r_cov_ref, l2_r_mu_ref, l2_r_cov_ref, l3_r_mu_ref, l3_r_cov_ref = self.model(l1_x, l1_y, l2_x, l2_y, l3_x, l3_y, x_ref, l1_y_ref, l2_y_ref, l3_y_ref, False)

            l1_nll_loss, l2_nll_loss, l3_nll_loss = self._compute_nll_loss(l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l3_output_mu, l3_output_cov, l1_truth, l2_truth, l3_truth)
            l1_kld_loss, l2_kld_loss, l3_kld_loss = self._compute_kld_loss(l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c, l3_z_mu_all, l3_z_cov_all, l3_z_mu_c, l3_z_cov_c)
            global_dist_loss = self._compute_global_dist_loss(l1_r_mu_ref, l1_r_cov_ref, l2_r_mu_ref, l2_r_cov_ref, l3_r_mu_ref, l3_r_cov_ref)
            loss = l1_nll_loss + l2_nll_loss + self.fidelity_weight * l3_nll_loss + l1_kld_loss + l2_kld_loss + l3_kld_loss + global_dist_loss
            # loss = l1_nll_loss + l2_nll_loss + l1_kld_loss + l2_kld_loss

            self._logger.debug(loss.item())

            losses.append(loss.item())
            l1_nll_losses.append(l1_nll_loss.item())
            l2_nll_losses.append(l2_nll_loss.item())
            l3_nll_losses.append(l3_nll_loss.item())
            l1_kld_losses.append(l1_kld_loss.item())
            l2_kld_losses.append(l2_kld_loss.item())
            l3_kld_losses.append(l3_kld_loss.item())
            global_dist_losses.append(global_dist_loss.item())

            # batches_seen += 1
            loss.backward()

            # gradient clipping - this does it in place
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()


            lr_scheduler.step()

            # _, _, val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)
            end_time = time.time()

            # self._writer.add_scalar('training loss',
            #                         np.mean(losses),
            #                         batches_seen)

            log_every = test_every_n_epochs
            if (epoch_num % log_every) == log_every - 1:
                self._logger.info("epoch complete")
                self._logger.info("evaluating now!")
                message = 'Epoch [{}/{}] train_loss: {:.4f}, l1_nll: {:.4f}, l1_kld: {:.4f}, l2_nll: {:.4f}, l2_kld: {:.4f}, l3_nll: {:.4f}, l3_kld: {:.4f}, global_dist: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs,
                                           np.mean(losses), np.mean(l1_nll_losses), np.mean(l1_kld_losses), np.mean(l2_nll_losses), np.mean(l2_kld_losses), np.mean(l3_nll_losses), np.mean(l3_kld_losses), np.mean(global_dist_losses), lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                l1_test_nll, l1_test_rmse, l1_test_nrmse, l2_test_nll, l2_test_rmse, l2_test_nrmse, l3_test_nll, l3_test_rmse, l3_test_nrmse, l3_y_truths_scaled, l3_y_preds_mu_scaled = self.evaluate(dataset='test', l1_z_mu_all=l1_z_mu_all, l1_z_cov_all=l1_z_cov_all, l2_z_mu_all=l2_z_mu_all, l2_z_cov_all=l2_z_cov_all, l3_z_mu_all=l3_z_mu_all, l3_z_cov_all=l3_z_cov_all)
                message = 'Epoch [{}/{}] test_l1_nll: {:.4f}, l1_rmse: {:.4f}, l1_nrmse: {:.4f}, l2_nll: {:.4f}, l2_rmse: {:.4f}, l2_nrmse: {:.4f},  l3_nll: {:.4f}, l3_rmse: {:.4f}, l3_nrmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs,
                                           l1_test_nll, l1_test_rmse, l1_test_nrmse, l2_test_nll, l2_test_rmse, l2_test_nrmse, l3_test_nll, l3_test_rmse, l3_test_nrmse, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

        return l1_z_mu_all, l1_z_cov_all, l2_z_mu_all, l2_z_cov_all, l3_z_mu_all, l3_z_cov_all, l1_test_nll, l1_test_rmse, l1_test_nrmse, l2_test_nll, l2_test_rmse, l2_test_nrmse, l3_test_nll, l3_test_rmse, l3_test_nrmse, l3_y_truths_scaled, l3_y_preds_mu_scaled

    def _test_l1_prepare_data(self, x, y):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.reshape(-1,self.input_dim)
        y = y.reshape(-1,self.l1_output_dim)

        return x.to(self.device), y.to(self.device)

    def _test_l2_prepare_data(self, x, y):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.reshape(-1,self.input_dim)
        y = y.reshape(-1,self.l2_output_dim)

        return x.to(self.device), y.to(self.device)

    def _test_l3_prepare_data(self, x, y):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.reshape(-1,self.input_dim)
        y = y.reshape(-1,self.l3_output_dim)

        return x.to(self.device), y.to(self.device)

    def _train_l1_prepare_data(self, x, y):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x = x.reshape(-1,self.input_dim)
        y = y.reshape(-1,self.l1_output_dim)
        return x.to(self.device), y.to(self.device)

    def _train_l2_prepare_data(self, x, y):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.reshape(-1,self.input_dim)
        y = y.reshape(-1,self.l2_output_dim)

        return x.to(self.device), y.to(self.device)

    def _train_l3_prepare_data(self, x, y):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.reshape(-1,self.input_dim)
        y = y.reshape(-1,self.l3_output_dim)

        return x.to(self.device), y.to(self.device)

    def _ref_prepare_data(self, x, l1_y, l2_y, l3_y):
        x = torch.from_numpy(x).float()
        l1_y = torch.from_numpy(l1_y).float()
        l2_y = torch.from_numpy(l2_y).float()
        l3_y = torch.from_numpy(l3_y).float()
        x = x.reshape(-1,self.input_dim)
        l1_y = l1_y.reshape(-1,self.l1_output_dim)
        l2_y = l2_y.reshape(-1,self.l2_output_dim)
        l3_y = l3_y.reshape(-1,self.l3_output_dim)

        return x.to(self.device), l1_y.to(self.device), l2_y.to(self.device), l3_y.to(self.device)


    def _compute_nll_loss(self, l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l3_output_mu, l3_output_cov, l1_truth, l2_truth, l3_truth):
        

        return nll_loss(l1_output_mu, l1_output_cov, l1_truth), nll_loss(l2_output_mu, l2_output_cov, l2_truth), nll_loss(l3_output_mu, l3_output_cov, l3_truth)

    def _compute_kld_loss(self, l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c, l3_z_mu_all, l3_z_cov_all, l3_z_mu_c, l3_z_cov_c):

        return kld_gaussian_loss(l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c), kld_gaussian_loss(l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c), kld_gaussian_loss(l3_z_mu_all, l3_z_cov_all, l3_z_mu_c, l3_z_cov_c)

    def _compute_global_dist_loss(self, l1_r_mu_ref, l1_r_cov_ref, l2_r_mu_ref, l2_r_cov_ref, l3_r_mu_ref, l3_r_cov_ref):

        l1_z_mu = torch.zeros(l1_r_mu_ref[0].shape).to(self.device)
        l1_z_cov = torch.ones(l1_r_cov_ref[0].shape).to(self.device)

        l2_z_mu = torch.zeros(l2_r_mu_ref[0].shape).to(self.device)
        l2_z_cov = torch.ones(l2_r_cov_ref[0].shape).to(self.device)

        l3_z_mu = torch.zeros(l3_r_mu_ref[0].shape).to(self.device)
        l3_z_cov = torch.ones(l3_r_cov_ref[0].shape).to(self.device)

        l1_v = l1_r_mu_ref - l1_z_mu
        l1_w_cov_inv = 1 / l1_r_cov_ref
        l1_z_cov_new = 1 / (1 / l1_z_cov + torch.sum(l1_w_cov_inv, dim=0))
        l1_z_mu_new = l1_z_mu + l1_z_cov_new * torch.sum(l1_w_cov_inv * l1_v, dim=0)

        l2_v = l2_r_mu_ref - l2_z_mu
        l2_w_cov_inv = 1 / l2_r_cov_ref
        l2_z_cov_new = 1 / (1 / l2_z_cov + torch.sum(l2_w_cov_inv, dim=0))
        l2_z_mu_new = l2_z_mu + l2_z_cov_new * torch.sum(l2_w_cov_inv * l2_v, dim=0)

        l3_v = l3_r_mu_ref - l3_z_mu
        l3_w_cov_inv = 1 / l3_r_cov_ref
        l3_z_cov_new = 1 / (1 / l3_z_cov + torch.sum(l3_w_cov_inv, dim=0))
        l3_z_mu_new = l3_z_mu + l3_z_cov_new * torch.sum(l3_w_cov_inv * l3_v, dim=0)

        js_loss_12 = 0.5 * (kld_gaussian_loss(l1_z_mu_new, l1_z_cov_new, l2_z_mu_new, l2_z_cov_new) + kld_gaussian_loss(l2_z_mu_new, l2_z_cov_new, l1_z_mu_new, l1_z_cov_new))
        js_loss_13 = 0.5 * (kld_gaussian_loss(l1_z_mu_new, l1_z_cov_new, l3_z_mu_new, l3_z_cov_new) + kld_gaussian_loss(l3_z_mu_new, l3_z_cov_new, l1_z_mu_new, l1_z_cov_new))
        js_loss_23 = 0.5 * (kld_gaussian_loss(l2_z_mu_new, l2_z_cov_new, l3_z_mu_new, l3_z_cov_new) + kld_gaussian_loss(l3_z_mu_new, l3_z_cov_new, l2_z_mu_new, l2_z_cov_new))
        js_loss = js_loss_12 + js_loss_13 + js_loss_23

        return js_loss
        

        

    def _test_loss(self, l1_y_preds_mu, l1_y_preds_cov, l1_y_truths, l2_y_preds_mu, l2_y_preds_cov, l2_y_truths, l3_y_preds_mu, l3_y_preds_cov, l3_y_truths):

        l1_nll = nll_metric(l1_y_preds_mu, l1_y_preds_cov, l1_y_truths)
        l2_nll = nll_metric(l2_y_preds_mu, l2_y_preds_cov, l2_y_truths)
        l3_nll = nll_metric(l3_y_preds_mu, l3_y_preds_cov, l3_y_truths)

        l1_y_truths_scaled = self.l1_y_scaler.inverse_transform(l1_y_truths)
        l1_y_preds_mu_scaled = self.l1_y_scaler.inverse_transform(l1_y_preds_mu)
        l1_std = self.l1_y_scaler.std
        l1_rmse = rmse_metric(l1_y_preds_mu_scaled, l1_y_truths_scaled)
        l1_nrmse = rmse_metric(l1_y_preds_mu_scaled, l1_y_truths_scaled)/l1_std

        l2_y_truths_scaled = self.l2_y_scaler.inverse_transform(l2_y_truths)
        l2_y_preds_mu_scaled = self.l2_y_scaler.inverse_transform(l2_y_preds_mu)
        l2_std = self.l2_y_scaler.std
        l2_rmse = rmse_metric(l2_y_preds_mu_scaled, l2_y_truths_scaled)
        l2_nrmse = rmse_metric(l2_y_preds_mu_scaled, l2_y_truths_scaled)/l2_std

        l3_y_truths_scaled = self.l3_y_scaler.inverse_transform(l3_y_truths)
        l3_y_preds_mu_scaled = self.l3_y_scaler.inverse_transform(l3_y_preds_mu)
        l3_std = self.l3_y_scaler.std
        l3_rmse = rmse_metric(l3_y_preds_mu_scaled, l3_y_truths_scaled)
        l3_nrmse = rmse_metric(l3_y_preds_mu_scaled, l3_y_truths_scaled)/l3_std

        return l1_nll, l1_rmse, l1_nrmse, l2_nll, l2_rmse, l2_nrmse, l3_nll, l3_rmse, l3_nrmse, l3_y_truths_scaled, l3_y_preds_mu_scaled



