import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Optimizer
import torch.nn.parallel
import numpy as np
import scipy.sparse as sparse
import torchvision.models
from opt.SGD import SGD

import random
import math


class gSGD():

    def __init__(self, models, lr,args, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        self.models = models
        self.lr_0 = self.lr_t = lr
        self.params = list(models.parameters())
        self.args = args
        self.ec_layer = self.cal_ec_layer()
        self.mask = self.cal_mask(models)
        self.cal_internal_optim()
        self.internal_optim.cal_ec_layer(self.ec_layer)
        # print([x[0] for x in list(models.named_parameters())])

    def cal_ec_layer(self):
        if self.args.ec_layer_type==0:
            ec_layer = self.models.ec_layer
        elif self.args.ec_layer_type == 1:
            ec_layer = self.cal_ec_layer_type1()
        return ec_layer

    def cal_ec_layer_type1(self):
        named_params = self.models.named_parameters()
        ec_layer = [[]]
        for layer_idx, (ii,jj) in enumerate(named_params):

            if   'weight' in ii :
                ec_layer[0].append(layer_idx)
        return ec_layer

    def cal_internal_optim(self):
        self.internal_optim = SGD(self.models.parameters(), lr=self.lr_t )

    def cal_mask(self, model):
        self.num_layer = len(self.params)
        self.num_ec_block = len(self.ec_layer)
        mask = []
        
        s_idx=[]
        s_h=[]
        for block in self.ec_layer:
            layer_mask = []
            layer_s_idx =[]
            layer_s_h = []
            num_layer_in_block = len(block)
            block_s_mask_idx = None
            tmp = [0,0]
            for ii in range(num_layer_in_block - 1):
                h = self.params[block[ii]].shape[0]
                if h > tmp[1]:
                    tmp = [ii, h]

            layer_s_idx = tmp[0]  # sub layer index (output size max)
            layer_s_h = tmp[1]  # max output size

            for ec_layer_idx, layer_idx in enumerate(block):
                layer = self.params[layer_idx]
                outcome, income = layer.shape
                if ec_layer_idx == 0:
                    loc = 'f'
                elif layer_idx == num_layer_in_block - 1:
                    loc = 'l'
                else:
                    loc = 'm'

                mask_tmp = self.generate_eye(outcome, income, loc).to(self.args.device)
                layer.data = layer.data * (1 - mask_tmp) + mask_tmp
                layer_mask.append(mask_tmp)

            mask.append(layer_mask)
            s_h.append(layer_s_h)
            s_idx.append(layer_s_idx)
        self.s_h = s_h
        self.s_idx = s_idx
        return (mask)


    def recover_s_layer(self,value,idx , shape):
        assert value.device == idx.device


        if value.device.type == 'cpu':
            return torch.sparse.FloatTensor(idx, value , shape).to_dense().to(self.args.device)
        else:

            return torch.cuda.sparse.FloatTensor(idx, value , shape).to_dense().to(self.args.device)

    def generate_eye(self, out_shape, in_shape, loc='m'):

        if loc == 'f':
            ratio = out_shape // in_shape + 1

            out_idx = list(range(out_shape))
            in_idx = list(range(in_shape )) *ratio
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(out_shape),
                    shape=[out_shape , in_shape ])
            )
        elif loc == 'l':


            ratio = in_shape // out_shape + 1

            out_idx = list(range(out_shape))*ratio
            in_idx = list(range(in_shape ))
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(in_shape),
                    shape=[out_shape , in_shape ])
            )
        elif  loc == 'm' :
            if in_shape > out_shape:
                ratio = in_shape // out_shape + 1

                out_idx = list(range(out_shape))*ratio
                in_idx = list(range(in_shape ))

            else:

                ratio = out_shape // in_shape + 1

                out_idx = list(range(out_shape))
                in_idx = list(range(in_shape)) * ratio


            idx_tmp = list(zip(out_idx, in_idx))

            idx = torch.LongTensor([x  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(max(out_shape, in_shape)),
                    shape=[out_shape , in_shape ])
            )

    def cal_R(self, lr, w_red, dw_red, sigmadwd, v_value):
        return (1 - lr * (dw_red  * w_red  - sigmadwd) / (v_value * v_value))

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def abs_model(self):
        for p in self.params:
            p = p.abs()

    def remain_input(self, x):
        return x.sum(0)

    def remain_output(self, x):
        return x.sum(1)

    def lr_decay(self, decay, epoch, max_epoch, decay_state):
        if decay == 'poly':
            lr = self.lr_0 * (1 - epoch / max_epoch) ** decay_state['power']
        elif decay == 'linear':
            lr = self.lr_0 * (1 - (epoch + 1) / max_epoch)
        elif decay == 'multistep':
            if (epoch + 1) in decay_state['step']:
                lr = self.lr_t * decay_state['gamma']
            else:
                lr = self.lr_t
        elif decay == 'exp':
            lr = self.lr_0 * math.e ** (-decay_state['power'] * epoch)
        elif decay == 'none':
            lr = self.lr_0
        for param_group in self.internal_optim.param_groups:
            param_group['lr'] = lr
        self.lr_t = lr


    def step(self,closure=None):
        if self.args.opt == "ec":
            self.ec_step(closure)
            self.bp_partial_step(closure)
        elif self.args.opt == 'bp':
            self.bp_step(closure)


    def bp_partial_step(self, closure):
        self.internal_optim.partial_bp_step_w()

    def bp_step(self,closure):
        self.internal_optim.step()


    def ec_step(self, closure=None):
        """Performs a single optimization step."""
        lr = self.lr_t

        for block_idx, block in enumerate(self.ec_layer):
            this_block_mask = self.mask[block_idx]
            this_block_s_h = self.s_h[block_idx]
            this_block_s_idx = self.s_idx[block_idx]
            # print(this_block_s_h)
            sigmadwd = torch.zeros(this_block_s_h).to(self.args.device)
            num_layer_in_block=len(block)

            for ec_layer_idx, layer_idx in enumerate(block):

                this_mask = this_block_mask[ec_layer_idx]
                layer = self.params[layer_idx]
                if ec_layer_idx == 0:
                    loc = 'f'
                elif ec_layer_idx == num_layer_in_block - 1:
                    loc = 'l'
                else:
                    loc = 'm'
                # print(ec_layer_idx, this_block_s_idx, '!!!!!!!!!!')
                if ec_layer_idx != this_block_s_idx:

                    w_blue =  layer.data * (1 - this_mask)
                    dw_blue = layer.grad.data * (1 - this_mask)

                    if ec_layer_idx < this_block_s_idx:
                        sigmadwd[:w_blue.shape[0]] +=  (w_blue * dw_blue).sum(1) ## remain output
                    elif ec_layer_idx > this_block_s_idx:
                        sigmadwd[:w_blue.shape[1]] +=  (w_blue * dw_blue).sum(0) ## remain input
                else:
                    v_value = self.remain_output(layer.data * this_mask)
                    w_red = v_value
                    dw_red = self.remain_output(layer.grad.data * this_mask)

            R = self.cal_R(lr=lr,
                           w_red=w_red,
                           dw_red=dw_red,
                           sigmadwd=sigmadwd,
                           v_value=v_value
                           )
            for ec_layer_idx, layer_idx in enumerate(block):
                this_mask = this_block_mask[ec_layer_idx]
                layer = self.params[layer_idx]
                if ec_layer_idx == 0:
                    loc = 'f'
                elif ec_layer_idx == num_layer_in_block - 1:
                    loc = 'l'
                else:
                    loc = 'm'

                if ec_layer_idx == this_block_s_idx:
                    layer_is_s = True
                else:
                    layer_is_s = False

                if layer_is_s:
                    this_R_value = (R).unsqueeze(1)
                    # print(layer.data.shape, R.shape)
                    layer.data = (layer.data - lr * layer.grad.data) * (1 - this_mask) + \
                                 layer.data * this_R_value * this_mask

                elif ec_layer_idx > this_block_s_idx:
                    out_shape, in_shape = layer.data.shape
                    layer.data = (layer.data -  lr * layer.grad.data / (v_value[:layer.data.shape[1]].view(1, -1) ** 2) ) / (R[:in_shape].view(1, -1)) * (1 - this_mask)+ \
                        layer.data * this_mask

                elif ec_layer_idx < this_block_s_idx:
                    out_shape, in_shape = layer.data.shape

                    layer.data = (layer.data -  lr * layer.grad.data / (v_value[:layer.data.shape[0]].view(-1, 1) ** 2) )/ (R[:out_shape].view(-1, 1)) *(1-this_mask) + \
                        layer.data * this_mask


