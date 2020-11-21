"""NICE model
"""

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

def _build_relu_network(latent_dim, hidden_dim, hidden_layers):
    _modules = [ nn.Linear(latent_dim, hidden_dim) ]
    for _ in range(hidden_layers):
        _modules.append( nn.Linear(hidden_dim, hidden_dim) )
        _modules.append( nn.ReLU() )
        _modules.append( nn.BatchNorm1d(hidden_dim) )
    _modules.append( nn.Linear(hidden_dim, latent_dim) )
    return nn.Sequential( *_modules )

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,in_out_dim, hidden_dim, hidden_layers,device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
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

        elif coupling_type == 'adaptive':

            v=0 #TODO >> Finish

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


def logistic_nice_loglkhd(h, diag):
    # ===== ===== Loss Function Implementations ===== =====
    """
    We assume that we final output of the network are components of a multivariate distribution that
    factorizes, i.e. the output is (y1,y2,...,yK) ~ p(Y) s.t. p(Y) = p_1(Y1) * p_2(Y2) * ... * p_K(YK),
    with each individual component's prior distribution coming from a standardized family of
    distributions, i.e. p_i == Gaussian(mu,sigma) for all i in 1..K, or p_i == Logistic(mu,scale).

    Definition of log-likelihood function with a Logistic prior.

    Args:
    * h: float tensor of shape (N,D). First dimension is batch dim, second dim consists of components
      of a factorized probability distribution.
    * diag: scaling diagonal of shape (D,).
    Returns:
    * loss: torch float tensor of shape (N,).

    """
    # \sum^D_i s_{ii} - { \sum^D_i log(exp(h)+1) + torch.log(exp(-h)+1) }
    return (torch.sum(diag.scale.data) - (torch.sum(torch.log1p(torch.exp(h[0])) + torch.log1p(torch.exp(-h[0])), dim=1)))

# wrap above loss functions in Modules:

class LogisticPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(LogisticPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag):
        if self.size_average:
            return torch.mean(-logistic_nice_loglkhd(fx, diag))
        else:
            return torch.sum(-logistic_nice_loglkhd(fx, diag))

###############################

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        #TODO fill in

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        #TODO fill in

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
