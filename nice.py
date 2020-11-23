"""NICE model
"""
# WORKING VERSION NO AFF COUP

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform,SigmoidTransform,AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F


"""Additive coupling layer.
"""

_get_even = lambda xs: xs[:,0::2]
_get_odd = lambda xs: xs[:,1::2]

def _build_relu_network(latent_dim, hidden_dim, hidden_layers, affine=False):
    _modules = nn.ModuleList([ nn.Linear(latent_dim, hidden_dim) ])
    _modules.append( nn.ReLU() )
    for _ in range(hidden_layers):
        _modules.append( nn.Linear(hidden_dim, hidden_dim) )
        _modules.append( nn.ReLU() )
    if affine == False:
        _modules.append( nn.Linear(hidden_dim, latent_dim) )
    else:
        _modules.append( nn.Linear(hidden_dim, int(latent_dim*2)) )
    return nn.Sequential(*_modules)

def _interleave(first, second, mask_config):
    """
    Given 2 rank-2 tensors with same batch dimension, interleave their columns.
    The tensors "first" and "second" are assumed to be of shape (B,M) and (B,N)
    where M = N or N+1, respectively.
    B is the batch size

    """
    cols = []
    if mask_config == 0:    #even
        for k in range(second.shape[1]):
            cols.append(first[:, k])
            cols.append(second[:, k])
        if first.shape[1] > second.shape[1]:
            cols.append(first[:, -1])
    else:   #odd
        for k in range(first.shape[1]):
            cols.append(second[:, k])
            cols.append(first[:, k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:, -1])
    return torch.stack(cols, dim=1)

class AdditiveCoupling(nn.Module):

    def __init__(self, in_out_dim, mask_config, nonlinearity):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()

        self.in_out_dim = in_out_dim
        self.mask_config = mask_config

        if mask_config:
            self._first = _get_odd
            self._second = _get_even

        else:
            self._first = _get_even
            self._second = _get_odd

        self.add_module('nonlinearity', nonlinearity)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """

        if reverse:
            y = _interleave(
                self._first(x),
                self.anticoupling_law(self._second(x), self.nonlinearity(self._first(x))),
                self.mask_config
            )

        else:
            y = _interleave(
                self._first(x),
                self.coupling_law(self._second(x),self.nonlinearity(self._first(x))),
                self.mask_config
            )
        return y, log_det_J

    def coupling_law(self, a, b):
        return (a + b)

    def anticoupling_law(self, a, b):
        return (a - b)

class AffineCoupling(nn.Module):
    
    def __init__(self, in_out_dim, mask_config, nonlinearity):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()

        self.in_out_dim = in_out_dim
        self.mask_config = mask_config

        if mask_config:
            self._first = _get_odd
            self._second = _get_even

        else:
            self._first = _get_even
            self._second = _get_odd

        self.add_module('nonlinearity', nonlinearity)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
            
        """
    

        if reverse:
            
            z1, z2 = torch.chunk(x, 2, dim=1)
            h = self.nonlinearity(z2)
            shift = h[:, 0::2]
            scale = torch.exp(h[:, 1::2])
            ya = (z1 - shift) / scale
            yb = z2
            log_det_J -= torch.log(scale).view(x.shape[0],-1).sum(-1)
            y = torch.cat([ya, yb], dim=1)
    
            
        else:
            
            z1, z2 = torch.chunk(x, 2, dim=1)
            h = self.nonlinearity(z2)
            shift = h[:, 0::2]
            scale = torch.exp(h[:, 1::2])
            ya = z1 * scale + shift
            yb = z2
            y = torch.cat([ya, yb], dim=1)
            log_det_J += torch.log(scale).view(x.shape[0],-1).sum(-1)

        return y, log_det_J


"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale) + self.eps
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J
    
class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__()

    def log_prob(self, x):
        """Computes data log-likelihood.
        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(F.softplus(x) + F.softplus(-x))
    
    def sample(self, size):
        
        """Samples from the distribution.
        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(0., 1.).sample(size).cuda()
        return torch.log(z) - torch.log(1. - z)

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,in_out_dim, hidden_dim, hidden_layers,device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'affine'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            hidden_dim: number of units in a hidden layer.
            hidden_layers: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        self.prior = prior
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type
        half_dim = int(in_out_dim / 2)
        if coupling_type == 'additive':
            odd = 1
            even = 0
            self.layer1 = AdditiveCoupling(in_out_dim, odd, _build_relu_network(half_dim, hidden_dim, hidden_layers))
            self.layer2 = AdditiveCoupling(in_out_dim, even, _build_relu_network(half_dim, hidden_dim, hidden_layers))
            self.layer3 = AdditiveCoupling(in_out_dim, odd, _build_relu_network(half_dim, hidden_dim, hidden_layers))
            self.layer4 = AdditiveCoupling(in_out_dim, even, _build_relu_network(half_dim, hidden_dim, hidden_layers))
            self.scaling_diag = Scaling(in_out_dim)

            # randomly initialize weights:
            for p in self.layer1.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
            for p in self.layer2.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
            for p in self.layer3.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
            for p in self.layer4.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)

        elif coupling_type == 'affine':
            odd = 1
            even = 0
            affineBool = True
            self.layer1 = AffineCoupling(in_out_dim, odd, _build_relu_network(half_dim, hidden_dim, hidden_layers,affineBool))
            self.layer2 = AffineCoupling(in_out_dim, even, _build_relu_network(half_dim, hidden_dim, hidden_layers,affineBool))
            self.layer3 = AffineCoupling(in_out_dim, odd, _build_relu_network(half_dim, hidden_dim, hidden_layers,affineBool))
            self.layer4 = AffineCoupling(in_out_dim, even, _build_relu_network(half_dim, hidden_dim, hidden_layers,affineBool))
            self.scaling_diag = Scaling(in_out_dim)

            # randomly initialize weights:
            for p in self.layer1.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
            for p in self.layer2.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
            for p in self.layer3.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
            for p in self.layer4.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
        else:
            raise ValueError('Coupling Type Error.')

    def f_inv(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        with torch.no_grad():
            x, _ = self.scaling_diag(z, reverse = True)
            x, _ = self.layer4(x, 0, reverse=True)
            x, _ = self.layer3(x, 0, reverse=True)
            x, _ = self.layer2(x, 0, reverse=True)
            x, _ = self.layer1(x, 0, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        log_det_J = 0

        y, log_det_J = self.layer1(x, log_det_J)
        y, log_det_J = self.layer2(y, log_det_J)
        y, log_det_J = self.layer3(y, log_det_J)
        y, log_det_J = self.layer4(y, log_det_J)
        y = self.scaling_diag(y)

        return y

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256)*self.in_out_dim #log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)

        return self.f_inv(z)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x).to(self.device)
    
