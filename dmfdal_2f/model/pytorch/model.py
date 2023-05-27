import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda:5")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLP_Encoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()

    def forward(self, x):
        # x = torch.cat([x,adj],dim=-1)
        output = self.model(x)
        mean = self.mean_out(output)
        # cov = self.cov_m(self.cov_out(output))
        cov = 0.1+ 0.9*self.cov_m(self.cov_out(output))
        return mean, cov

class MLP_Decoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim, 
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Softplus()

    def forward(self, x):
        # x = torch.cat([x,adj],dim=-1)
        output = self.model(x)
        mean = self.mean_out(output)
        cov = self.cov_m(self.cov_out(output))
        # cov = torch.exp(self.cov_out(output))

        return mean, cov


class Model(nn.Module):
    def __init__(self, logger, **model_kwargs):
        super().__init__()
        self.device = torch.device(model_kwargs.get('device')) #"cuda:5"
        self.hidden_layers = int(model_kwargs.get('hidden_layers',2))
        self.z_dim = int(model_kwargs.get('z_dim',32))
        self.input_dim = int(model_kwargs.get('input_dim', 3))
        self.l1_output_dim = int(model_kwargs.get('l1_output_dim', 256))
        self.l2_output_dim = int(model_kwargs.get('l2_output_dim', 1024))
        self.hidden_dim = int(model_kwargs.get('hidden_dim', 32))
        self.encoder_output_dim = self.z_dim
        self.decoder_input_dim = self.z_dim + self.input_dim
        self.context_percentage_low = float(model_kwargs.get('context_percentage_low', 0.2))
        self.context_percentage_high = float(model_kwargs.get('context_percentage_high', 0.5))

        self.l1_encoder_model_local = MLP_Encoder(self.input_dim+self.l1_output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l1_encoder_model_global = MLP_Encoder(self.input_dim+self.l1_output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_encoder_model_local = MLP_Encoder(self.input_dim+self.l2_output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_encoder_model_global = MLP_Encoder(self.input_dim+self.l2_output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)

        self.l1_decoder_model = MLP_Decoder(self.decoder_input_dim, self.l1_output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_decoder_model = MLP_Decoder(self.decoder_input_dim, self.l2_output_dim, self.hidden_layers, self.hidden_dim)

        # self.z2_z1_agg = MLP_Z1Z2_Encoder(self.z_dim, self.z_dim)

        self._logger = logger


    def split_context_target(self, x, y, context_percentage_low, context_percentage_high):
        """Helper function to split randomly into context and target"""
        context_percentage = np.random.uniform(context_percentage_low,context_percentage_high)
        # if level == 1:
        #     node_dim = 18
        # elif level == 2:
        #     node_dim = 85
        # x = x.reshape(-1,node_dim,x.shape[-1])
        # y = y.reshape(-1,node_dim,y.shape[-1])
        # adj= adj.reshape(-1,node_dim,node_dim)

        n_context = int(x.shape[0]*context_percentage)
        ind = np.arange(x.shape[0])
        mask = np.random.choice(ind, size=n_context, replace=False)
        others = np.delete(ind,mask)


        return x[mask], y[mask], x[others], y[others]


    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(n,var.size(0)).normal_()).to(self.device)

        std = torch.sqrt(var)
        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def xy_to_r_local(self, x, y, level):
        if level == 1:
            r_mu, r_cov = self.l1_encoder_model_local(torch.cat([x, y],dim=-1))
        elif level == 2:
            r_mu, r_cov = self.l2_encoder_model_local(torch.cat([x, y],dim=-1))

        return r_mu, r_cov

    def xy_to_r_global(self, x, y, level):
        if level == 1:
            r_mu, r_cov = self.l1_encoder_model_global(torch.cat([x, y],dim=-1))
        elif level == 2:
            r_mu, r_cov = self.l2_encoder_model_global(torch.cat([x, y],dim=-1))

        return r_mu, r_cov

    def z_to_y(self, x, zs, level):

        # outputs = []

        if level == 1:
            output = self.l1_decoder_model(torch.cat([x,zs], dim=-1))

        elif level == 2:
            output = self.l2_decoder_model(torch.cat([x,zs], dim=-1))

        return output

    def ba_z_agg(self, r_mu, r_cov):

        # r_mu = torch.swapaxes(r_mu,0,1)
        # r_cov = torch.swapaxes(r_cov,0,1)
        z_mu = torch.zeros(r_mu[0].shape).to(self.device)
        z_cov = torch.ones(r_cov[0].shape).to(self.device)

        # r_mu = torch.cat([r_mu_k, r_mu_g],0)
        # r_cov = torch.cat([r_cov_k, r_cov_g],0)

        v = r_mu - z_mu
        w_cov_inv = 1 / r_cov
        z_cov_new = 1 / (1 / z_cov + torch.sum(w_cov_inv, dim=0))
        z_mu_new = z_mu + z_cov_new * torch.sum(w_cov_inv * v, dim=0)
        return z_mu_new, z_cov_new


    def forward(self, l1_x_all=None, l1_y_all=None, l2_x_all=None, l2_y_all=None, x_ref=None, l1_y_ref=None, l2_y_ref=None, test=False, l1_x_test=None, l2_x_test=None, l1_z_mu_all=None, l1_z_cov_all=None, l2_z_mu_all=None, l2_z_cov_all=None):

        if test==False:
            self._logger.debug("starting point complete, starting split source and target")
            #first half for context, second for target
            l1_x_c,l1_y_c,l1_x_t,l1_y_t = self.split_context_target(l1_x_all,l1_y_all, self.context_percentage_low, self.context_percentage_high)
            l2_x_c,l2_y_c,l2_x_t,l2_y_t = self.split_context_target(l2_x_all,l2_y_all, self.context_percentage_low, self.context_percentage_high)
            self._logger.debug("data split complete, starting encoder")

            # compute ref distance
            # print('x_ref.shape, l1_y_ref.shape', x_ref.shape, l1_y_ref.shape)
            l1_r_mu_ref, l1_r_cov_ref = self.xy_to_r_global(x_ref, l1_y_ref, level=1)
            l2_r_mu_ref, l2_r_cov_ref = self.xy_to_r_global(x_ref, l2_y_ref, level=2)

            #l1_encoder
            l1_r_mu_all_k, l1_r_cov_all_k = self.xy_to_r_local(l1_x_all, l1_y_all, level=1)
            l1_r_mu_c_k, l1_r_cov_c_k = self.xy_to_r_local(l1_x_c, l1_y_c, level=1)
            l1_r_mu_all_g, l1_r_cov_all_g = self.xy_to_r_global(l1_x_all, l1_y_all, level=1)
            l1_r_mu_c_g, l1_r_cov_c_g = self.xy_to_r_global(l1_x_c, l1_y_c, level=1)

            #l2_encoder
            l2_r_mu_all_k, l2_r_cov_all_k = self.xy_to_r_local(l2_x_all, l2_y_all, level=2)
            l2_r_mu_c_k, l2_r_cov_c_k = self.xy_to_r_local(l2_x_c, l2_y_c, level=2)
            l2_r_mu_all_g, l2_r_cov_all_g = self.xy_to_r_global(l2_x_all, l2_y_all, level=2)
            l2_r_mu_c_g, l2_r_cov_c_g = self.xy_to_r_global(l2_x_c, l2_y_c, level=2)

            l1_r_mu_all = torch.cat([l1_r_mu_all_k, l1_r_mu_all_g, l2_r_mu_all_g],0)
            l2_r_mu_all = torch.cat([l2_r_mu_all_k, l1_r_mu_all_g, l2_r_mu_all_g],0)
            l1_r_cov_all = torch.cat([l1_r_cov_all_k, l1_r_cov_all_g, l2_r_cov_all_g],0)
            l2_r_cov_all = torch.cat([l2_r_cov_all_k, l1_r_cov_all_g, l2_r_cov_all_g],0)

            l1_r_mu_c = torch.cat([l1_r_mu_c_k, l1_r_mu_c_g, l2_r_mu_all_g],0)
            l2_r_mu_c = torch.cat([l2_r_mu_c_k, l1_r_mu_all_g, l2_r_mu_c_g],0)
            l1_r_cov_c = torch.cat([l1_r_cov_c_k, l1_r_cov_c_g, l2_r_cov_all_g],0)
            l2_r_cov_c = torch.cat([l2_r_cov_c_k, l1_r_cov_all_g, l2_r_cov_c_g],0)
            
            l1_z_mu_all, l1_z_cov_all = self.ba_z_agg(l1_r_mu_all, l1_r_cov_all)
            l1_z_mu_c, l1_z_cov_c = self.ba_z_agg(l1_r_mu_c, l1_r_cov_c)

            l2_z_mu_all, l2_z_cov_all = self.ba_z_agg(l2_r_mu_all, l2_r_cov_all)
            l2_z_mu_c, l2_z_cov_c = self.ba_z_agg(l2_r_mu_c, l2_r_cov_c)

            #sample z
            l1_zs = self.sample_z(l1_z_mu_all, l1_z_cov_all, l1_x_t.size(0))
            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_t.size(0))
            #l1_decoder, l2_decoder
            self._logger.debug("Encoder complete, starting decoder")
            l1_output_mu, l1_output_cov = self.z_to_y(l1_x_t,l1_zs, level=1)
            l2_output_mu, l2_output_cov = self.z_to_y(l2_x_t,l2_zs, level=2)

            l1_truth = l1_y_t
            l2_truth = l2_y_t

            self._logger.debug("Decoder complete")
            # if batches_seen == 0:
            #     self._logger.info(
            #         "Total trainable parameters {}".format(count_parameters(self))
            #     )

            return l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l1_truth, l2_truth, l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c, l1_r_mu_ref, l1_r_cov_ref, l2_r_mu_ref, l2_r_cov_ref
            
        else:
            # l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov = None, None, None, None
            # l1_r_mu_all_k1, l1_r_cov_all_k1 = self.xy_to_r_local(l1_x_all, l1_y_all, level=1)
            # l1_r_mu_all_g1, l1_r_cov_all_g1 = self.xy_to_r_global(l1_x_all, l1_y_all, level=1)
            # l2_r_mu_all_k2, l2_r_cov_all_k2 = self.xy_to_r_local(l2_x_all, l2_y_all, level=2)
            # l2_r_mu_all_g2, l2_r_cov_all_g2 = self.xy_to_r_global(l2_x_all, l2_y_all, level=2)

            # l1_z_mu_all, l1_z_cov_all = self.ba_z_agg(l1_r_mu_all_k1, l1_r_cov_all_k1, l1_r_mu_all_g1, l1_r_cov_all_g1)
            # l2_z_mu_all, l2_z_cov_all = self.ba_z_agg(l2_r_mu_all_k2, l2_r_cov_all_k2, l2_r_mu_all_g2, l2_r_cov_all_g2)

            # l1_r_mu_all, l1_r_cov_all = self.xy_to_r(l1_x_all, l1_y_all, level=1)

            # if l1_x_all is not None:
            l1_zs = self.sample_z(l1_z_mu_all, l1_z_cov_all, l1_x_test.size(0))
            l1_output_mu, l1_output_cov = self.z_to_y(l1_x_test, l1_zs, level=1)
            # if l2_x_all is not None:
            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_test.size(0))
            l2_output_mu, l2_output_cov = self.z_to_y(l2_x_test, l2_zs, level=2)


            return l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov


