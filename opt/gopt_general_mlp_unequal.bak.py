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
        self.expand = None
        self.ec_layer = self.models.ec_layer
        self.mask = self.cal_mask(models)
        self.cal_internal_optim()
        self.internal_optim.cal_ec_layer(self.ec_layer)

    def cal_internal_optim(self):
        self.internal_optim = SGD(self.models.parameters(), lr=self.lr_t )

    def expand_layer_mask(self,mask):
        outcome,income = mask.shape
        expand = outcome - income
        if self.expand is None:
            self.expand = expand
        assert expand == self.expand
        tmp1 = mask[:income,:]
        tmp2 = mask[income:,:]
        tmp2[tmp2==1]=2

    def shrink_layer_mask(self, mask):
        outcome, income= mask.shape
        expand =  income -outcome
        if self.expand is None:
            self.expand = expand
        assert expand == self.expand
        tmp1 = mask[:,:outcome]
        tmp2 = mask[:,outcome:]
        tmp2[tmp2==1]=3

    def cal_mask(self, model):
        self.s_idx = 0
        self.num_layer = len(self.params)
        self.num_ec_layer = len(self.ec_layer)
        mask = []
        #



        self.s_idx = 0
        self.s_h = self.params[self.ec_layer[self.s_idx]].shape[0]

        for ec_layer_idx, layer_idx in enumerate(self.ec_layer):
            layer = self.params[layer_idx]
            outcome, income = layer.shape
            if ec_layer_idx == 0:
                loc = 'f'
            elif ec_layer_idx == self.num_ec_layer - 1:
                loc = 'l'
            else:
                loc = 'm'

            if loc == 'f'   :
                mask_tmp = self.generate_eye(outcome, income, loc).to(self.args.device)
            elif  loc == 'l':
                mask_tmp = self.generate_eye(outcome, income, loc).to(self.args.device)
            else:
                if outcome>income:
                    mask_tmp = self.generate_eye(outcome,income,loc).to(self.args.device)
                    self.expand_layer_mask(mask_tmp)
                elif outcome == income:
                    mask_tmp = self.generate_eye(outcome, income, loc).to(self.args.device)
                elif outcome < income:
                    mask_tmp = self.generate_eye(outcome, income, loc).to(self.args.device)
                    self.shrink_layer_mask(mask_tmp)

            mask.append(mask_tmp)
            this_mask_red =    (mask[-1]==1).to(torch.float32) +  (mask[-1]==3).to(torch.float32)   + (mask[-1]==2).to(torch.float32)
            layer.data = layer.data * (1 - this_mask_red) + this_mask_red
        # s_mask_idx = (mask[self.s_idx]!=0).nonzero().transpose(0,1)
        # s_mask_idx_shape = mask[self.s_idx].shape

            # torch.sparse.FloatTensor(a, torch.FloatTensor([1, 1, 1, 1])).to_dense()


        return (mask )


    def recover_s_layer(self,value,idx , shape):
        assert value.device == idx.device
        # ##print(idx, value)

        if value.device.type == 'cpu':
            return torch.sparse.FloatTensor(idx, value , shape).to_dense().to(self.args.device)
        else:
            # ##print(value.device, idx.devic)
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
            # ##print(len(idx_tmp) , len(in_idx) , len(out_idx), len(idx),max(out_shape, in_shape))
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(max(out_shape, in_shape)),
                    shape=[out_shape , in_shape ])
            )

    def compress_to(self, income, outcome, tmp):
        res = torch.zeros(outcome).to(self.args.device)
        ratio = income // outcome
        start = 0
        remain = income % outcome
        for ii in range(ratio):
            end = start + outcome
            res[:outcome] += tmp[start:end]
            start = end
        if remain >0  :
            res[:remain] += tmp[-remain:]
        return(res)


    def cal_R(self, lr, w_red, dw_red, sigmadwd, v_value):
        return (1 - lr * (dw_red  * w_red  - sigmadwd) / (v_value * v_value))

    def cal_delta_v_1(self, dw_i, w_j, ec_layer_idx):
        if ec_layer_idx ==1:
            res = dw_i * w_j.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        else:
            res = 0
        return res

    def cal_delta_v_6(self, dw_dash_red,w_dash_red, sigmadwiwi_k, ec_layer_idx, w_j ):
        dw_dash_red = dw_dash_red[-self.expand:]
        w_dash_red = w_dash_red[-self.expand:]
        res  = dw_dash_red*w_dash_red - sigmadwiwi_k[ec_layer_idx]
        ratio = self.expand // self.s_h + 1
        tmp_wj = w_j.repeat(ratio)
        tmp_wj = tmp_wj[:self.expand]
        res = res / tmp_wj / w_dash_red
        ##print(tmp_wj, w_dash_red, sigmadwiwi_k, dw_dash_red,w_dash_red,'7777777777777777777')

        # ##print(res.shape)
        return res


    def cal_delta_v_4(self, dw_j,sigmadwd_red, sigmadwd_dash_red, v_j, sigmadwi_wj):

        tmp_sigmadwi_wj = sigmadwi_wj.sum(0)
        tmp_sigmadwd_dash_red = sigmadwd_dash_red.sum(0)
        dim = v_j.shape[0]
        tmp_sigmadwd_dash_red = self.compress_to(self.expand, dim, tmp_sigmadwd_dash_red)
        tmp_sigmadwi_wj = self.compress_to(self.expand, dim, tmp_sigmadwi_wj)

        res = v_j * dw_j - sigmadwd_red  - tmp_sigmadwd_dash_red - tmp_sigmadwi_wj
        res = res / v_j
        return res

    def cal_delta_v_2(self,delta_wi, w_j, w_k ):
        return(delta_wi / w_j / w_k)

    def cal_R_2(self,lr, delta_v_2, v_2):
        # print('##############',  sum(delta_v_2==0), sum(v_2==0),   (delta_v_2 / v_2).sum(),  (1 - lr*delta_v_2 / v_2).max() , lr)
        return(1 - lr*delta_v_2 / v_2)

    def cal_R_4(self,lr,delta_v_4, v_j):
        return(1-lr* delta_v_4 /v_j)

    def cal_R_6(self, lr, delta_v_6, v_6 ):
        v_6 = v_6[-self.expand:]
        return(1-lr*delta_v_6/v_6)


    def have_update_red(self,loc, income, outcome):
        if loc == "m" and income<outcome:
            return True
        else:
            return False


    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()



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
            # ##print('bp')
            self.bp_step(closure)


    def bp_partial_step(self, closure):
        self.internal_optim.partial_bp_step_w()

    def bp_step(self,closure):
        self.internal_optim.step()


    def ec_step(self, closure=None):
        """Performs a single optimization step."""

        model = self.models
        num_ec_layer = len(self.ec_layer)
        mask = self.mask
        lr = self.lr_t
        # assert lr ==0

        w_red = []
        w_blue = []
        dw_red = []
        dw_blue = []
        v_j = w_j = torch.masked_select(self.params[self.ec_layer[0]].data , self.mask[0]==1)
        # ##print([[x[0],x[1] ] for x in enumerate( self.params)])
        # ##print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # ##print(self.params[self.ec_layer[0]].data )
        # assert 1==0
        dw_j = torch.masked_select(self.params[self.ec_layer[0]].grad.data , self.mask[0]==1)



        sigmadwd_red = torch.zeros( self.s_h).to(self.args.device)
        # sigmadwd_red_all = torch.zeros(num_ec_layer,  self.s_h).to(self.args.device)
        sigmadwd_dash_red = torch.zeros(num_ec_layer, self.expand).to(self.args.device)
        # sigmadwi_wj = torch.zeros(num_ec_layer, self.expand ).to(self.args.device)
        sigmadwiwi_wj = torch.zeros(num_ec_layer, self.expand ).to(self.args.device)
        # sigmadwi  = torch.zeros(num_ec_layer, self.expand).to(self.args.device)
        sigmadwiwi  = torch.zeros(num_ec_layer, self.expand).to(self.args.device)
        # v_6 = torch.zeros(num_ec_layer,self.expand).to(self.args.device)





        for ec_layer_idx, layer_idx in enumerate(self.ec_layer):

            layer = self.params[layer_idx]
            outcome, income = layer.shape
            if ec_layer_idx == 0:
                loc = 'f'
            elif ec_layer_idx == self.num_ec_layer - 1:
                loc = 'l'
            else:
                loc = 'm'



            layer_up = self.params[layer_idx+1]
            this_mask = mask[ec_layer_idx]

            this_mask_blue = (this_mask==0).to(torch.float32)
            this_mask_red = (this_mask==1).to(torch.float32)
            this_mask_dash_red_update =  (this_mask==2).to(torch.float32)
            this_mask_dash_red_noupdate = (this_mask==3).to(torch.float32)



            w_dash_red = self.remain_output(layer.data * this_mask_dash_red_update)
            dw_dash_red = self.remain_output(layer.grad.data * this_mask_dash_red_update)


            if loc =='f':
                pass
            elif loc =='m':
                pass
            else:
                pass

            if self.have_update_red(loc=loc, income=income, outcome=outcome):

                # ##print(ec_layer_idx, len(mask))
                up_mask_blue = (mask[ec_layer_idx + 1] == 0).to(torch.float32)

                ## w_k
                w_k_idx = this_mask_dash_red_update.nonzero()
                num_k = len(w_k_idx)
                assert num_k == self.expand
                ## for each k, compute the sum of dw_i, sum of w_i* dw_i
                # ##print(layer_up.grad.data * up_mask_blue)
                # assert 1==2
                # sigmadwi[ec_layer_idx]  =  self.remain_input((layer_up.grad.data * up_mask_blue)[:,w_k_idx[:,0]])
                sigmadwiwi[ec_layer_idx]  =  self.remain_input(
                    (layer_up.grad.data * layer_up.data* up_mask_blue)[:,w_k_idx[:,0]])

                # for each k, compute the sum of w_i /w_j
                if self.expand <= income:
                    tmp_sigmadwiwi_wj =sigmadwiwi[ec_layer_idx] / w_j[:self.expand]
                else:
                    ratio = self.expand // income +1
                    tmp_wj = w_j[:income].repeat(ratio)
                    tmp_wj = tmp_wj[:self.expand]
                    tmp_sigmadwiwi_wj = sigmadwiwi[ec_layer_idx] / tmp_wj
                    # ##print(w_j, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # assert 1==2

                sigmadwiwi_wj[ec_layer_idx] = tmp_sigmadwiwi_wj
                ##print(sigmadwiwi[ec_layer_idx], 99999999999999,layer_up.grad.data,layer_up.data,up_mask_blue,w_k_idx[:,0])

                ## for each k compute the sum of w_k according to the output (k layer)

                sigmadwd_dash_red[ec_layer_idx] =   self.remain_output(layer.grad.data *layer.data * this_mask_dash_red_update )[-self.expand:]

            ## compute sigmadwd_red(every layer)
            if loc is not 'f':
                tmp_sigmadwd = self.remain_input(layer.grad.data *layer.data * this_mask_blue)
                if ec_layer_idx ==1:
                    sigmadwd_red += tmp_sigmadwd

                elif income>outcome:
                    ratio = income // outcome
                    start = 0
                    remain = income%outcome
                    for ii in range(ratio):
                        end = start + outcome
                        sigmadwd_red[:outcome] += tmp_sigmadwd[start:end]
                        start = end

                    if remain >0 :
                        sigmadwd_red[:remain] += tmp_sigmadwd[-remain:]
                else:
                    sigmadwd_red[:income] = sigmadwd_red[:income] + tmp_sigmadwd

        delta_v_6 =[]
        R_6 = []
        for ec_layer_idx, layer_idx in enumerate(self.ec_layer):

            layer = self.params[layer_idx]
            outcome, income = layer.shape
            if ec_layer_idx == 0:
                loc = 'f'
            elif ec_layer_idx == self.num_ec_layer - 1:
                loc = 'l'
            else:
                loc = 'm'


            layer_up = self.params[layer_idx+1]
            this_mask = mask[ec_layer_idx]

            this_mask_blue = (this_mask==0).to(torch.float32)
            this_mask_red = (this_mask==1).to(torch.float32)
            this_mask_dash_red_update =  (this_mask==2).to(torch.float32)
            this_mask_dash_red_noupdate = (this_mask==3).to(torch.float32)



            if loc == 'f':
                delta_v_4 = self.cal_delta_v_4(dw_j= dw_j,
                                                   sigmadwd_red=sigmadwd_red,     ### sigma w_i* dw_i  through w_j
                                                   sigmadwd_dash_red=sigmadwd_dash_red,### sigma w_k* dw_k
                                                   v_j= v_j,
                                                   sigmadwi_wj=sigmadwiwi_wj)
                R_4 = self.cal_R_4(lr=lr, delta_v_4=delta_v_4, v_j=v_j)

                # ##print(R_4, '!!!!!!!','delta_v_4',delta_v_4, 'sigmadwd_red',sigmadwd_red,'sigmadwd_dash_red',sigmadwd_dash_red,'v_j',v_j, 'sigmadwiwi_wj', sigmadwiwi_wj)

            if self.have_update_red(loc=loc, income=income, outcome=outcome):
                w_dash_red = self.remain_output(layer.data * this_mask_dash_red_update)
                dw_dash_red = self.remain_output(layer.grad.data * this_mask_dash_red_update)
                ##print(layer.data , this_mask_dash_red_update, 8888888888888888888888, w_dash_red)

                delta_v_6.append(self.cal_delta_v_6(dw_dash_red= dw_dash_red,
                                               w_dash_red= w_dash_red ,
                                            sigmadwiwi_k=sigmadwiwi,
                                               ec_layer_idx=ec_layer_idx,
                                               w_j = w_j))
                R_6.append(self.cal_R_6(lr=lr,
                                        delta_v_6=delta_v_6[-1],
                                        v_6= w_j * w_dash_red
                                        ))
            else:
                delta_v_6.append([])
                R_6.append([])




        for ec_layer_idx, layer_idx in enumerate(self.ec_layer):


            layer = self.params[layer_idx]
            outcome, income = layer.shape
            if ec_layer_idx == 0:
                loc = 'f'
            elif ec_layer_idx == self.num_ec_layer - 1:
                loc = 'l'
            else:
                loc = 'm'


            layer_up = self.params[layer_idx+1]
            this_mask = mask[ec_layer_idx]

            this_mask_blue = (this_mask==0).to(torch.float32)
            this_mask_red = (this_mask==1).to(torch.float32)
            this_mask_dash_red_update =  (this_mask==2).to(torch.float32)
            this_mask_dash_red_noupdate = (this_mask==3).to(torch.float32)


            if loc =='f':
                ## 45
                this_R_4 = R_4.unsqueeze(1)
                ##print(this_R_4, '1111111111111111111')
                # assert 1==2
                if (this_R_4 != this_R_4).sum() >1:
                    assert 1==2

                tmp_blue =  (
                                layer.data - lr * layer.grad.data
                            ) * this_mask_blue
                tmp_red = (
                            layer.data*this_R_4
                            ) * this_mask_red


                # print((layer.data - tmp_blue - tmp_red).sum(),1)
                layer.data = tmp_blue + tmp_red
                


            elif   self.have_update_red(loc, income,outcome):
                ## 6 13


                tmp_red = (
                              layer.data
                          ) * this_mask_red

                this_R_6 = R_6[ec_layer_idx].unsqueeze(1)
                ratio = self.expand // income + 1
                tmp_R_4 = R_4.repeat(ratio)

                this_R_4 = tmp_R_4[:income]
                this_R_4 = this_R_4.unsqueeze(0)

                ##print(this_R_6, this_R_4,  '22222222222222222', R_6)


                tmp_update_dash_red = (
                    layer.data
                )*this_mask_dash_red_update
                tmp_update_dash_red[-self.expand:,:] = tmp_update_dash_red[-self.expand:,:] * this_R_6 / this_R_4



                ratio = self.expand //  income  + 1
                tmp_R_4 = R_4.repeat(ratio)
                tmp_v_4 = v_j.repeat(ratio)
                this_R_4 = tmp_R_4[:income]
                this_v_4 = tmp_v_4[:income]
                tmp_blue = (
                                   (layer.data - lr * layer.grad.data / this_v_4) / this_R_4
                           ) * this_mask_blue

                # print((layer.data - tmp_blue - tmp_red - tmp_update_dash_red).sum(),2)
                layer.data = tmp_blue + tmp_update_dash_red + tmp_red


            elif not self.have_update_red(loc, income,outcome):
                ##13 2
                tmp_red = (
                              layer.data
                          ) * this_mask_red

                tmp_noupdate_dash_red = (
                    layer.data
                )*this_mask_dash_red_noupdate

                if ec_layer_idx ==1:
                    this_R_4 = R_4.unsqueeze(0)
                    this_v_4 = v_j.unsqueeze(0)
                    ##print(this_R_4, this_v_4,'333333333333333333')
                    tmp_blue = (
                                       (layer.data - lr * layer.grad.data / this_v_4) / this_R_4
                               ) * this_mask_blue
                    # assert  (tmp_blue != (layer.data * this_mask_blue)).sum()==0

                else:
                    if loc != 'l':
                        assert income > outcome
                    # if ec_layer_idx == 5:
                    #     ##print(ec_layer_idx, tmp_noupdate_dash_red, 'tmp_red',  tmp_red)

                    this_v_4 = v_j[:income].unsqueeze(0)
                    this_R_4 = R_4[:income].unsqueeze(0)



                    tmp_blue_1 = (
                                       (layer.data - lr * layer.grad.data / this_v_4) / this_R_4
                               ) #* this_mask_blue
                    # tmp_blue_1[:,-self.expand:]=0

                    ratio = self.expand // outcome + 1
                    tmp_R_4 = R_4[:outcome].repeat(ratio)
                    tmp_v_4 = v_j[:outcome].repeat(ratio)
                    this_R_4 = tmp_R_4[:self.expand]
                    this_v_4 = tmp_v_4[:self.expand]

                    this_R_6 = R_6[ec_layer_idx-1].unsqueeze(0)


                    w_k = torch.masked_select(self.params[layer_idx-1], self.mask[ec_layer_idx -1]==2)

                    ##print(  this_R_4,this_R_6, w_k, '44444444444444444444')

                    this_v_4.unsqueeze_(0)
                    w_k.unsqueeze_(0)

                    tmp_delta_v_2 = self.cal_delta_v_2(delta_wi= layer.grad.data[:,-self.expand:] , #outcome x self.expand
                                       w_j= this_v_4,                                               # 1 x self.expand
                                       w_k=  w_k )                                                  # 1 x self.expand

                    ###           outcome x self.expand         1xself.expand       1x self.expand
                    # ##print(w_k   , this_v_4, '$$$$$$$$$$$$')
                    v_2 = layer.data[ :, -self.expand:] *   w_k               *  this_v_4
                    ### outcome x self.expand
                    tmp_R_2 = self.cal_R_2(lr=lr, delta_v_2=tmp_delta_v_2, v_2=v_2)

                    tmp_blue_2 = tmp_R_2 * layer.data[ :, -self.expand:] / this_R_4 / this_R_6

                    # ##print([x.shape for x in [tmp_blue_2,  tmp_blue_1[:,-self.expand].data, this_R_6, this_R_6]], 111)

                    tmp_blue_1[:,-self.expand:] = tmp_blue_2

                    tmp_blue = tmp_blue_1 *  this_mask_blue
                    tmp_blue = tmp_blue.data
                    # print( sum(layer.grad.data[ :, -self.expand:] ==0) , 4)
                    # ##print('R2',tmp_R_2.data.numpy(), 'R6',this_R_6.data.numpy(), 'R4',this_R_4.data.numpy())
                # print((layer.data - tmp_blue - tmp_red - tmp_noupdate_dash_red).sum(),3)
                # print(tmp_blue.sum(), tmp_red.sum(), tmp_noupdate_dash_red.sum(),3)
                # print(layer.data.sum() , 3)
                layer.data = tmp_blue + tmp_red + tmp_noupdate_dash_red








