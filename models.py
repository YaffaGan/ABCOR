import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import ipdb

class COR(nn.Module):
    '''
    item feature is not used in this model, 
    '''
    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims, \
                                                     item_feature, adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_dims = mlp_p1_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs
        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # not used in this model, extended for using item feature in future work
        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim], \
                                               requires_grad=True).cuda()

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_dims = self.mlp_p1_dims[:-1] + [self.mlp_p1_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p1_dims[:-1], temp_p1_dims[1:])])
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)
        
        self.init_weights()

    def reuse_Z2(self,D,E1):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)
        mu, _ = self.encode(torch.cat((D,E1),1))
        h_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]
        return h_p2
        
    def forward(self, D, E1, Z2_reuse=None, CI=0):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D) # meituan and yelp
        encoder_input = torch.cat((D, E1), 1)  # D,E1
        mu, logvar = self.encode(encoder_input)    # E2 distribution
        E2 = self.reparameterize(mu, logvar)       # E2
        
        if CI == 1: # D=NULL
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)
            mu_0, logvar_0 = self.encode(encoder_input_0)
            E2_0 = self.reparameterize(mu_0, logvar_0)
            scores = self.decode(E1, E2, E2_0, Z2_reuse)
        else:
            scores = self.decode(E1, E2, None, Z2_reuse)
        reg_loss = self.reg_loss()
        return scores, mu, logvar, reg_loss

    def encode(self, encoder_input):
        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h) 
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, E1, E2, E2_0=None, Z2_reuse=None):

        if E2_0 is None:
            h_p1 = torch.cat((E1, E2), 1)
        else:
            h_p1 = torch.cat((E1, E2_0), 1)
            
        for i, layer in enumerate(self.mlp_p1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                Z1_mu = h_p1[:, :self.mlp_p1_dims[-1]]
                Z1_logvar = h_p1[:, self.mlp_p1_dims[-1]:]

        h_p2 = E2
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]

        if Z2_reuse!=None:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = Z2_reuse 
        else:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2 = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                    Z2 = torch.unsqueeze(Z2, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2_ = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z2_ = torch.unsqueeze(Z2_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
                    Z2 = torch.cat([Z2,Z2_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = torch.mean(Z2, 0)

        user_preference = torch.cat((Z1, Z2), 1)


        h_p3 = user_preference
        for i, layer in enumerate(self.mlp_p3_layers):
            h_p3 = layer(h_p3)
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)
        return h_p3

    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for layer in self.mlp_p3_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss


class ABCORV(nn.Module):
    '''
    item feature is not used in this model, 
    '''
    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims, \
                                                     item_feature, adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(ABCORV, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_dims = mlp_p1_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs
        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # not used in this model, extended for using item feature in future work
        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim], \
                                               requires_grad=True).cuda()

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_dims = self.mlp_p1_dims[:-1] + [self.mlp_p1_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p1_dims[:-1], temp_p1_dims[1:])])
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)
        
        self.init_weights()
        #change2:实例化注意力
        self.attentionp2 = Attention(attention_type='dot',hidden_size=temp_p2_dims[-1])
        #self.attentionp2 = Attention(attention_type='cosine',hidden_size=temp_p2_dims[-1])
        #self.attentionp2 = Attention(attention_type='mlp',hidden_size=temp_p2_dims[-1])


    def reuse_Z2(self,D,E1):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)
        mu, _,keyh = self.encode(torch.cat((D,E1),1))
        attn_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            attn_p2 = layer(attn_p2)
            if i != len(self.mlp_p2_layers) - 1:
                attn_p2 = self.act_function(attn_p2)
            else:
                padded_mu = F.pad(mu,(0,keyh.shape[1]-mu.shape[1]),'constant',0)
                length =int(keyh.shape[0]*keyh.shape[1]/attn_p2.shape[0]/attn_p2.shape[1])
                keyh = keyh.reshape(attn_p2.shape[0],length,attn_p2.shape[1])
                padded_mu = padded_mu.reshape(attn_p2.shape[0],length, int(padded_mu.shape[0]*padded_mu.shape[1]/attn_p2.shape[0]/length))
                attn_padded_mu, _ = self.attentionp2(attn_p2,keyh,padded_mu)

                attn_mu = attn_padded_mu[:,:,:int(attn_padded_mu.shape[2]/2)].squeeze(1)
                    #print(attn_E2.shape)
                    
                if mu.shape[1]%attn_mu.shape[1]==0:
                    attn_temp_mu = attn_mu
                    for i in range(int(mu.shape[1]/attn_mu.shape[1])-1):
                        attn_mu = torch.cat((attn_mu, attn_temp_mu), dim=1)
        h_p2 = attn_mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]
        return h_p2
        
    def forward(self, D, E1, Z2_reuse=None, CI=0):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D) # meituan and yelp
        encoder_input = torch.cat((D, E1), 1)  # D,E1
        mu, logvar, keyh = self.encode(encoder_input)    # E2 distribution  #change4
        E2 = self.reparameterize(mu, logvar)       # E2
        
        if CI == 1: # D=NULL
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)
            mu_0, logvar_0, keyh_0 = self.encode(encoder_input_0) #change5  #Change-1
            E2_0 = self.reparameterize(mu_0, logvar_0)
            scores = self.decode(E1, E2, keyh, E2_0, keyh_0,Z2_reuse)  #change6 #Change-2
        else:
            scores = self.decode(E1, E2, keyh, None, None, Z2_reuse)  #change7 #Change-3
        reg_loss = self.reg_loss()
        return scores, mu, logvar, reg_loss

    def encode(self, encoder_input):
        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h) 
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]
        return mu, logvar,h #change3

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, E1, E2, keyh, E2_0=None, keyh_0=None, Z2_reuse=None): #change8 #Change-4

        if E2_0 is None:
            attn_p2 = E2
            for i, layer in enumerate(self.mlp_p2_layers):
                attn_p2 = layer(attn_p2)
                if i != len(self.mlp_p2_layers) - 1:
                    attn_p2 = self.act_function(attn_p2)
                else:

                    if keyh.shape[1]%attn_p2.shape[1]!=0:
                            temp_dim_attn = (int(keyh.shape[1]/attn_p2.shape[1])+1)*attn_p2.shape[1]
                            padded_keyh = F.pad(keyh,(0,temp_dim_attn-keyh.shape[1]),'constant',0)
                    else:
                        padded_keyh = keyh
                
                    padded_E2 = F.pad(E2,(0,padded_keyh.shape[1]-E2.shape[1]),'constant',0)
                    beishu = padded_E2.shape[1]/E2.shape[1]
                    length =int(padded_keyh.shape[0]*padded_keyh.shape[1]/attn_p2.shape[0]/attn_p2.shape[1])
                    padded_keyh = padded_keyh.reshape(attn_p2.shape[0],length,attn_p2.shape[1])
                    padded_E2 = padded_E2.reshape(attn_p2.shape[0],length, int(padded_E2.shape[0]*padded_E2.shape[1]/attn_p2.shape[0]/length))
                    attn_padded_E2, _ = self.attentionp2(attn_p2,padded_keyh,padded_E2)
                    attn_E2 = attn_padded_E2[:,:,:int(attn_padded_E2.shape[2]/beishu)].squeeze(1)

                    if E2.shape[1]%attn_E2.shape[1]==0:
                        attn_temp_E2 = attn_E2
                        for i in range(int(E2.shape[1]/attn_E2.shape[1])-1):
                            attn_E2 = torch.cat((attn_E2, attn_temp_E2), dim=1) 
                    #print(attn_E2.shape)
        else:
            attn_p2 = E2
            attn_p2_0 = E2_0
            for i, layer in enumerate(self.mlp_p2_layers):
                attn_p2 = layer(attn_p2)
                attn_p2_0 = layer(attn_p2_0)
                if i != len(self.mlp_p2_layers) - 1:
                    attn_p2 = self.act_function(attn_p2)
                    attn_p2_0 = self.act_function(attn_p2_0)
                else:
                    if keyh.shape[1]%attn_p2.shape[1]!=0:
                            temp_dim_attn = (int(keyh.shape[1]/attn_p2.shape[1])+1)*attn_p2.shape[1]
                            padded_keyh = F.pad(keyh,(0,temp_dim_attn-keyh.shape[1]),'constant',0)
                    else:
                        padded_keyh = keyh
                    padded_E2 = F.pad(E2,(0,padded_keyh.shape[1]-E2.shape[1]),'constant',0)
                    beishu = padded_E2.shape[1]/E2.shape[1]
                    length =int(padded_keyh.shape[0]*padded_keyh.shape[1]/attn_p2.shape[0]/attn_p2.shape[1])
                    padded_keyh = padded_keyh.reshape(attn_p2.shape[0],length,attn_p2.shape[1])
                    padded_E2 = padded_E2.reshape(attn_p2.shape[0],length, int(padded_E2.shape[0]*padded_E2.shape[1]/attn_p2.shape[0]/length))
                    attn_padded_E2, _ = self.attentionp2(attn_p2,padded_keyh,padded_E2)
                    attn_E2 = attn_padded_E2[:,:,:int(attn_padded_E2.shape[2]/beishu)].squeeze(1)
                    #print(attn_E2.shape)
                    
                    if E2.shape[1]%attn_E2.shape[1]==0:
                        attn_temp_E2 = attn_E2
                        for i in range(int(E2.shape[1]/attn_E2.shape[1])-1):
                            attn_E2 = torch.cat((attn_E2, attn_temp_E2), dim=1)

                    #print(attn_E2.shape)

                    if keyh_0.shape[1]%attn_p2_0.shape[1]!=0:
                            temp_dim_attn_0 = (int(keyh_0.shape[1]/attn_p2_0.shape[1])+1)*attn_p2_0.shape[1]
                            padded_keyh_0 = F.pad(keyh_0,(0,temp_dim_attn_0-keyh_0.shape[1]),'constant',0)
                    else:
                        padded_keyh_0 = keyh_0
                    
                    padded_E2_0 = F.pad(E2_0,(0,padded_keyh_0.shape[1]-E2_0.shape[1]),'constant',0) #Change-5
                    beishu_0 = padded_E2_0.shape[1]/E2_0.shape[1]
                    length_0 =int(padded_keyh_0.shape[0]*padded_keyh_0.shape[1]/attn_p2_0.shape[0]/attn_p2_0.shape[1]) #Change-6
                    padded_keyh_0 = padded_keyh_0.reshape(attn_p2_0.shape[0],length_0,attn_p2_0.shape[1])   #Change-7
                    padded_E2_0 = padded_E2_0.reshape(attn_p2_0.shape[0],length_0, int(padded_E2_0.shape[0]*padded_E2_0.shape[1]/attn_p2_0.shape[0]/length_0)) #Change-8
                    attn_padded_E2_0, _ = self.attentionp2(attn_p2_0,padded_keyh_0,padded_E2_0) #Change-9
                    attn_E2_0 = attn_padded_E2_0[:,:,:int(attn_padded_E2_0.shape[2]/beishu_0)].squeeze(1)  #Change-10
                    
                    #Change-11
                    if E2_0.shape[1]%attn_E2_0.shape[1]==0:
                        attn_temp_E2_0 = attn_E2_0
                        for i in range(int(E2_0.shape[1]/attn_E2_0.shape[1])-1):
                            attn_E2_0 = torch.cat((attn_E2_0, attn_temp_E2_0), dim=1)
    
        if E2_0 is None:
            h_p1 = torch.cat((E1, attn_E2), 1) #change9
        else:
            h_p1 = torch.cat((E1, attn_E2_0), 1)  #Change-12
            
        for i, layer in enumerate(self.mlp_p1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                Z1_mu = h_p1[:, :self.mlp_p1_dims[-1]]
                Z1_logvar = h_p1[:, self.mlp_p1_dims[-1]:]

        h_p2 = attn_E2 #change10
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]

        if Z2_reuse!=None:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = Z2_reuse 
        else:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2 = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                    Z2 = torch.unsqueeze(Z2, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2_ = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z2_ = torch.unsqueeze(Z2_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
                    Z2 = torch.cat([Z2,Z2_], 0)
            Z1 = torch.mean(Z1, 0)
            Z2 = torch.mean(Z2, 0)
        

        user_preference = torch.cat((Z1, Z2), 1)

        h_p3 = user_preference 
        for i, layer in enumerate(self.mlp_p3_layers):
            h_p3 = layer(h_p3)
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)
        return h_p3

    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for layer in self.mlp_p3_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss


#change1:定义注意力类
class Attention(nn.Module):
    def __init__(self,attention_type='dot',hidden_size=256):
        super(Attention, self).__init__()
        self.attention_type = attention_type
        self.hidden_size = hidden_size

        #Linear layer to transform the query(decoder hidden state)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        #Linear layer to transform the key(encoder hidden state)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        #Linear layer to transform the vaule(encoder hidden state)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        #alpha_ij = V_a * tanh(W_ae * enc_i + W_ad * dec_j + b_a)
        #V_a
        self.to_scores = nn.Linear(hidden_size, 1, bias=False)

        #Softmax layer to compute the attention weights
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,query,keys,values):
        #Transform the query
        query = self.query(query).unsqueeze(1)
        #print(query.shape)
        #Transform the keys
        keys = self.key(keys)
        #print(keys.shape)
        #Transform the values
        values = self.value(values)
        #print(values.shape)

        #compute the attention weights
        if self.attention_type == 'dot':
            #dot product attention
            attention_weights = torch.bmm(query, keys.transpose(1,2))
        elif self.attention_type == 'cosine':
            #cosine similarity attention
            query = query / query.norm(dim=-1, keepdim=True)
            keys = keys / keys.norm(dim=-1, keepdim=True)
            attention_weights = torch.bmm(query, keys.transpose(1,2))
        elif self.attention_type == 'mlp':
            #MLP-based attention
            hidden_att = F.tanh(query + keys)
            attention_weights = self.to_scores(hidden_att).transpose(1,2)
        else:
            raise ValueError(f"Invaild attention type:{self.attention_type}")
        
        #Normalize the attention weights
        attention_weights = self.softmax(attention_weights)

        #Apply the attention_weights to the values to obtain the attended output
        attended_output = torch.bmm(attention_weights, values)

        return attended_output,attention_weights


class ABCORM(nn.Module):
    '''
    item feature is not used in this model, 
    '''
    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims, \
                                                     item_feature, adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(ABCORM, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_dims = mlp_p1_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs
        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # not used in this model, extended for using item feature in future work
        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim], \
                                               requires_grad=True).cuda()

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_dims = self.mlp_p1_dims[:-1] + [self.mlp_p1_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p1_dims[:-1], temp_p1_dims[1:])])
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)
        
        self.init_weights()
        #change2:实例化注意力
        self.attentionz1 = MultiHeadSelfAttention(hidden_dim = self.mlp_p1_dims[-1], num_heads = 10)
        self.attentionz2 = MultiHeadSelfAttention(hidden_dim = self.mlp_p2_dims[-1], num_heads = 10)
    

    def reuse_Z2(self,D,E1):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)
        mu, _ = self.encode(torch.cat((D,E1),1))
        h_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]
        return h_p2
        
    def forward(self, D, E1, Z2_reuse=None, CI=0):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D) # meituan and yelp
        encoder_input = torch.cat((D, E1), 1)  # D,E1
        mu, logvar = self.encode(encoder_input)    # E2 distribution  #change4
        E2 = self.reparameterize(mu, logvar)       # E2
        
        if CI == 1: # D=NULL
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)
            mu_0, logvar_0 = self.encode(encoder_input_0) #change5  #Change-1
            E2_0 = self.reparameterize(mu_0, logvar_0)
            scores = self.decode(E1, E2, E2_0, Z2_reuse)  #change6 #Change-2
        else:
            scores = self.decode(E1, E2, None,  Z2_reuse)  #change7 #Change-3
        reg_loss = self.reg_loss()
        return scores, mu, logvar, reg_loss

    def encode(self, encoder_input):
        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h) 
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]
        return mu, logvar #change3

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, E1, E2, E2_0=None, Z2_reuse=None): #change8 #Change-4
    
        if E2_0 is None:
            h_p1 = torch.cat((E1, E2), 1) #change9
        else:
            h_p1 = torch.cat((E1, E2_0), 1)  #Change-12
            
        for i, layer in enumerate(self.mlp_p1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                Z1_mu = h_p1[:, :self.mlp_p1_dims[-1]]
                Z1_logvar = h_p1[:, self.mlp_p1_dims[-1]:]

        h_p2 = E2 #change10
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)
            else:
                Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]
                Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]

        if Z2_reuse!=None:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
            Z1 = torch.mean(attn_Z1, 0)
            Z2 = Z2_reuse
        else:
            for i in range(self.sample_freq):
                if i == 0:
                    Z1 = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2 = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1 = torch.unsqueeze(Z1, 0)
                    Z2 = torch.unsqueeze(Z2, 0)
                else:
                    Z1_ = self.reparameterize(Z1_mu, Z1_logvar)
                    Z2_ = self.reparameterize(Z2_mu, Z2_logvar)
                    Z1_ = torch.unsqueeze(Z1_, 0)
                    Z2_ = torch.unsqueeze(Z2_, 0)
                    Z1 = torch.cat([Z1,Z1_], 0)
                    Z2 = torch.cat([Z2,Z2_], 0)

            Z1 = torch.mean(Z1, 0)
            Z2 = torch.mean(Z2, 0)  
           
        Z1_res = Z1.reshape(1, Z1.shape[0],int(Z1.shape[1]))
        attn_Z1 = self.attentionz1(Z1_res)
        attn_Z1 = attn_Z1.squeeze(0)
        Z2_res = Z2.reshape(1, Z2.shape[0],int(Z2.shape[1]))
        attn_Z2 = self.attentionz2(Z2_res)
        attn_Z2 = attn_Z2.squeeze(0)

        user_preference = torch.cat((attn_Z1, attn_Z2), 1)

        h_p3 = user_preference 
        for i, layer in enumerate(self.mlp_p3_layers):
            h_p3 = layer(h_p3)
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)
        return h_p3

    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for layer in self.mlp_p3_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss


#change1:定义注意力类
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim //num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size() 
        
        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
        #计算注意力权重
        attention_weights = torch.matmul(q, k.transpose(-2,-1))/torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        x = self.fc(output) + x

        return x


def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD