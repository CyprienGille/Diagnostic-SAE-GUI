# -*- coding: utf-8 -*-

# # 210 - First convolution network (CNN) with pytorch
#
# First convolution network on MNIST database.

# **Note:** install [tqdm](https://pypi.python.org/pypi/tqdm) if not installed: ``!pip install tqdm``


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import functions.functions_torch as ft

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm

# if 'float' in str(torch.get_default_dtype()):
#    torch.set_default_dtype(torch.double)

# torch.set_default_tensor_type(torch.DoubleTensor)
"""
This file provides a basic idea to construct the functions related to NN in torch
Besides the part of constructing the network, it has functions as following:
"""

__all__ = [
    "initialize_weights",
    "check_cuda_model",
    "apply_proj2model",
    "train",
    "train_novisu",
    "train_multiOpts",
    "train_novisu_multiOpts",
    "train_with_proj",
    "train_with_proj_novisu",
    "test",
    "weights_and_sparsity",
    "weights_and_sparsityByAxis",
    "show_sparsity",
    "visu_convLayer",
    "create_model_name",
    "create_model_name_MultiOpts",
    "save_net",
    "save_net_params",
    "load_net",
    "load_net_params",
    "run_loader",
    "run_loader_multiOpts",
    "run",
    "topk_error",
]
# In[2]:


BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
DATA_DIR = "../datas/"
USE_CUDA = False
N_EPOCHS = 10

# In[3]:
def initialize_weights(model, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():  # if usuing GPU
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


# In[4]:
# =======================================================================
# =                             Classes                                 =
# =======================================================================
# Part Net
#   Convolution Network
class Net_conv(nn.Module):
    def __init__(self, nb_class=10, p=0.5):
        super(Net_conv, self).__init__()
        # self.avg_pool  = nn.AdaptiveAvgPool2d((5,5))
        # =>
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(p=p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(50, nb_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)


class Net_conv_noDropout(nn.Module):  # No drop out
    def __init__(self, nb_class=10):
        super(Net_conv_noDropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 50), nn.ReLU(inplace=True), nn.Linear(50, nb_class),
        )

    def convlayer_x(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.features(x)
        # batch_size * 20 *4 *4
        x = x.view(-1, 320)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)


class Net_conv_1(nn.Module):
    # Another way of define Net_conv for future use
    def __init__(self, nb_class=10, p=0.5):
        super(Net_conv_1, self).__init__()
        # self.avg_pool  = nn.AdaptiveAvgPool2d((5,5))
        # =>
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.lin1 = nn.Linear(320, 50)
        self.lin2 = nn.Linear(50, nb_class)
        self.drop2d = nn.Dropout2d(p=p)
        self.drop1d = nn.Dropout(p=p)

    def forward(self, x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)), (2, 2)
        )  # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.drop2d(self.conv2(x))), (2, 2))
        x = x.view(-1, 320)
        x = F.relu(self.drop1d(self.lin1(x)))
        x = F.relu(self.lin2(x))
        return F.log_softmax(x, dim=-1)


class Net_conv_noDropout_1(nn.Module):
    # Another way of define Net_conv_noDropout for future use
    def __init__(self, nb_class=10):
        super(Net_conv_noDropout_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.lin1 = nn.Linear(320, 50)
        self.lin2 = nn.Linear(50, nb_class)

    def convlayer_x(self, x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)), (2, 2)
        )  # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        return x

    def forward(self, x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)), (2, 2)
        )  # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, 320)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return F.log_softmax(x, dim=-1)


# LeNet-300-100
class LeNet_300_100(nn.Module):
    def __init__(self, nb_class=10, do_dropout=False, p=0.5):
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.out = nn.Linear(100, nb_class)
        if do_dropout:
            self.p = p
        else:
            self.p = None

    def forward(self, x):
        if x.dim() > 1:
            x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        if self.p is not None:
            x = F.dropout(x, p=self.p)
        x = F.relu(self.fc2(x))
        if self.p is not None:
            x = F.dropout(x, p=self.p)
        x = self.out(x)
        return F.log_softmax(x, dim=-1)


# =======================================================================
# =                           Functions                                 =
# =======================================================================
# In[5]:
# Part Functions (Train and test)
def check_cuda_model(model, USE_CUDA, N_EPOCHS):
    if USE_CUDA:
        try:
            model = model.cuda()
        except Exception as e:
            print(e)
            USE_CUDA = False
            N_EPOCHS = 2


def apply_proj2model(model, proj_func, params):
    """
    Apply the projection function *proj_func* to the Conv2d and Linear layer
    of the model.
    """

    lst_proj_type = ["proj_l1ball", "proj_l21ball", "proj_l12ball", "proj_nuclear"]
    if proj_func.__name__ not in lst_proj_type:
        raise ValueError(
            "Unknown projection type, expected \n{},\n got {}".format(
                lst_proj_type, proj_func.__name__
            )
        )
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # params for proj func
            try:
                eta = params["ETA"]
            except KeyError:
                eta = params["eta"]

            if proj_func.__name__ in ["proj_l1ball", "proj_nuclear"]:
                m.weight.data = proj_func(m.weight.data, eta).reshape(
                    m.weight.data.size()
                )
                if m.bias is not None:
                    m.bias.data = proj_func(m.bias.data, eta)
            elif proj_func.__name__ in [
                "proj_l21ball",
                "proj_l12ball",
            ]:
                try:
                    axis = params["AXIS"]
                except KeyError:
                    axis = params["axis"]
                m.weight.data = proj_func(m.weight.data, eta, axis).reshape(
                    m.weight.data.size()
                )
                if m.bias is not None:
                    m.bias.data = proj_func(m.bias.data, eta, axis)


def train(
    epoch, train_loader, model, optimizer, loss_func, keep_pbar=False, verbose=True
):
    # Train 1 epoch
    lst_opt = ["proximal", "graddz", "adagrad_dz"]
    model.train()
    losses = []
    loader = tqdm(train_loader, total=len(train_loader), desc="Epoch {}".format(epoch))
    for batch_idx, (data, target) in enumerate(loader):
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data.type(dtype=torch.get_default_dtype()))
        # cross-entropy loss
        # target = target.squeeze_()
        # target= torch.as_tensor(target,dtype=output.dtype)
        if any(x in loss_func._get_name().lower() for x in ["nll", "crossentropy"]):
            loss = loss_func(output, target)
        else:
            loss = loss_func(output, target.type(dtype=output.dtype))
        loss.backward()
        if any(str_x in optimizer.__class__.__name__.lower() for str_x in lst_opt):
            optimizer.step(epoch)
        else:
            optimizer.step()
        losses.append(float(loss.item()))
        if verbose and batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    if not keep_pbar:
        loader.close()
    return model, np.mean(losses)


def train_novisu(epoch, train_loader, model, optimizer, loss_func, verbose=True):
    # Train 1 epoch
    lst_opt = ["proximal", "graddz", "adagrad_dz"]
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        target = target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data.type(dtype=torch.get_default_dtype()))
        if any(x in loss_func._get_name().lower() for x in ["nll", "crossentropy"]):
            loss = loss_func(output, target)
        else:
            loss = loss_func(output, target.type(dtype=output.dtype))
        loss.backward()
        if any(str_x in optimizer.__class__.__name__.lower() for str_x in lst_opt):
            optimizer.step(epoch)
        else:
            optimizer.step()
        losses.append(float(loss.item()))
        if verbose and batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return model, np.mean(losses)


# ===== Train multi-opts
def train_multiOpts(
    epoch, train_loader, model, optimizers, loss_func, keep_pbar=False, verbose=True
):
    # Train 1 epoch for multiple optimizers (with progressbar)
    lst_opt = ["proximal", "graddz", "adagrad_dz"]
    model.train()
    losses = []
    loader = tqdm(train_loader, total=len(train_loader), desc="Epoch {}".format(epoch))
    for batch_idx, (data, target) in enumerate(loader):
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        for optimizer in optimizers:
            optimizer.zero_grad()
        output = model(data.type(dtype=torch.get_default_dtype()))
        # cross-entropy loss
        # target = target.squeeze_()
        # target= torch.as_tensor(target,dtype=output.dtype)
        if any(x in loss_func._get_name().lower() for x in ["nll", "crossentropy"]):
            loss = loss_func(output, target)
        else:
            loss = loss_func(output, target.type(dtype=output.dtype))
        loss.backward()
        for optimizer in optimizers:
            if any(str_x in optimizer.__class__.__name__.lower() for str_x in lst_opt):
                optimizer.step(epoch)
            else:
                optimizer.step()
        losses.append(float(loss.item()))
        if verbose and batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    if not keep_pbar:
        loader.close()
    return model, np.mean(losses)


def train_novisu_multiOpts(
    epoch, train_loader, model, optimizers, loss_func, verbose=True, log=[]
):
    # Train 1 epoch for multiple optimizers (without progressbar)
    lst_opt = ["proximal", "graddz", "adagrad_dz"]
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        target = target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        for optimizer in optimizers:
            optimizer.zero_grad()
        output = model(data.type(dtype=torch.get_default_dtype()))
        if any(x in loss_func._get_name().lower() for x in ["nll", "crossentropy"]):
            loss = loss_func(output, target)
        else:
            loss = loss_func(output, target.type(dtype=output.dtype))
        loss.backward()
        for optimizer in optimizers:
            log.append(
                optimizer.__class__.__name__
                + (
                    "(CS)"
                    if optimizer.state_dict()["param_groups"][0]["control_sparsity"]
                    else ""
                )
            )
            if any(str_x in optimizer.__class__.__name__.lower() for str_x in lst_opt):
                optimizer.step(epoch)
            else:
                optimizer.step()
        losses.append(float(loss.item()))
        if verbose and batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return model, np.mean(losses)


def train_with_proj(
    epoch,
    train_loader,
    model,
    optimizer,
    loss_func,
    proj_func,
    params,
    keep_pbar=False,
    verbose=True,
):
    # Train 1 epoch
    lst_opt = ["proximal", "graddz", "adagrad_dz"]
    model.train()
    losses = []
    loader = tqdm(train_loader, total=len(train_loader), desc="Epoch {}".format(epoch))
    for batch_idx, (data, target) in enumerate(loader):
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data.type(dtype=torch.get_default_dtype()))
        # cross-entropy loss
        # target = target.squeeze_()
        # target= torch.as_tensor(target,dtype=output.dtype)
        if any(x in loss_func._get_name().lower() for x in ["nll", "crossentropy"]):
            loss = loss_func(output, target)
        else:
            loss = loss_func(output, target.type(dtype=output.dtype))
        loss.backward()
        if any(str_x in optimizer.__class__.__name__.lower() for str_x in lst_opt):
            optimizer.step(epoch)
        else:
            optimizer.step()
        losses.append(float(loss.item()))

        apply_proj2model(model, proj_func, params)

        if verbose and batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    if not keep_pbar:
        loader.close()
    return model, np.mean(losses)


def train_with_proj_novisu(
    epoch, train_loader, model, optimizer, loss_func, proj_func, params, verbose=True
):
    # Train 1 epoch
    lst_opt = ["proximal", "graddz", "adagrad_dz"]
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        target = target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data.type(dtype=torch.get_default_dtype()))
        if any(x in loss_func._get_name().lower() for x in ["nll", "crossentropy"]):
            loss = loss_func(output, target)
        else:
            loss = loss_func(output, target.type(dtype=output.dtype))
        loss.backward()
        if any(str_x in optimizer.__class__.__name__.lower() for str_x in lst_opt):
            optimizer.step(epoch)
        else:
            optimizer.step()
        losses.append(float(loss.item()))

        apply_proj2model(model, proj_func, params)

        if verbose and batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return model, np.mean(losses)


# ====== Test
def test(test_loader, model, loss_func, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data.requires_grad_()
        output = model(data.type(dtype=torch.get_default_dtype()))
        if any(x in loss_func._get_name().lower() for x in ["nll", "crossentropy"]):
            test_loss += loss_func(output, target).item()
            _, pred = torch.max(
                output.data, 1
            )  # get the index of the max log-probability
            correct += (
                pred.eq(target.type(dtype=pred.dtype).data.view_as(pred))
                .cpu()
                .sum()
                .item()
            )
        else:
            test_loss += loss_func(
                output, target.type(dtype=output.dtype)
            ).item()  # sum up batch loss
            _, pred = torch.max(
                output.data, 1
            )  # get the index of the max log-probability
            # pred.add_(1)
            correct += (
                pred.eq(target.type(dtype=pred.dtype).data.view_as(pred))
                .cpu()
                .sum()
                .item()
            )

    test_loss /= len(test_loader.dataset)
    if verbose:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return [float(test_loss), correct]


# In[6]:
# Functions (Others)
def weights_and_sparsity(model, tol=1.0e-3):
    """
    It extracts the weights and calculate their spasity (using the tol as the threshold to judge zero element)
    respectively from the model, and return two dict type results.
    """
    state = model.state_dict()
    weights = {}
    spsty = {}
    for key in state.keys():
        if "weight" in key.lower():
            w = state[key].cpu().numpy()
            # w[w<tol]=0.0
            weights[key] = w
            spsty[key] = ft.sparsity(state[key], tol=tol)
    return weights, spsty


def weights_and_sparsityByAxis(model, tol=1.0e-3, axis=0):
    """
    It extracts the weights and calculate their spasity (using the tol as the threshold to judge zero element)
    by given axis (default is the first dimension) for the model, and return two dict type results.
    """
    state = model.state_dict()
    weights = {}
    spsty = {}
    for key in state.keys():
        if "weight" in key.lower():
            w = state[key].cpu()
            weights[key] = w.numpy()
            M = torch.where(torch.abs(w) < tol, torch.zeros_like(w), torch.abs(w))
            if len(M.size()) == 2:
                summ = M.sum(dim=axis).numpy()
            elif len(M.size()) == 4:
                # normally the weight matrix of a conv1d tensor will be n_out, n_in, kernal_size1, kernal_size2
                summ = M.sum(dim=3).sum(dim=2).sum(dim=1).numpy()
            else:
                continue
            ind = np.where(summ == 0.0)
            spsty[key] = len(ind[0]) / len(summ)
    return weights, spsty


def show_sparsity(
    weights,
    spasities,
    target_layers=[2, 3],
    tol=1.0e-03,
    show_title=False,
    ttl_tag="",
    cmap="jet",
    binary_map=False,
    concat_maps=True,
    keepfig=True,
    saveres=False,
    outputPath="../results/",
):
    """  
    Ít takes the output of function weights_and_sparsity
    and demonstrate the sparsity figure of the chosen layers
    (Default layers are 3rd and 4th)
        
    Return the spacity for the input matrix M                                   
    ----- INPUT                                                                 
        weights             : (dict) weight matrices of each layer
        spasities           : (dict) spasities of the weight matrices 
        target_layers       : (int,list, optional) list of interested layers, default: [2,3]
        tol                 : (scalar, optional) tolerance of zero element, default: 1.0e-3
        ttl_tag             : (string, optional) title tag for the graphs, default: ''
        cmap                : (string, optional) color map for the heatmap, see https://matplotlib.org/users/colormaps.html for more details
        binary_map          : (boolean, optional) whether choose to show the binary heatmap, default: False
        concat_maps         : (boolean, optional) whether to show the concatenation of the heatmap of last two layers,
                                default: True 
        keepfig             : (boolean, optional) whether kepp the existed figures, default: False
        saveres             : (boolean, optional) whether save the figures locally, default: False
        outputPath          : (str, optional) the directory to save the figures if saveres is True, default: '../results/'
 
    """
    import seaborn as sns
    import scipy.sparse as sparse

    if type(weights) is dict:
        w = list(weights.values())
    else:
        w = weights
    if type(spasities) is dict:
        spasity = list(spasities.values())
    else:
        spasity = spasities
    lst_ax = []
    lst_w = []
    lst_order = []
    if not keepfig:
        plt.close("all")

    if type(target_layers) is int:
        concat_maps = False
        fig = plt.figure()
        W0 = w[target_layers]
        W = np.abs(W0)
        ax = sns.heatmap(
            W, xticklabels=[0, W.shape[1]], yticklabels=[0, W.shape[0]], cmap=cmap
        )
        plt.xticks([0, W0.shape[1]])
        plt.yticks([0, W0.shape[0]])
        plt.tight_layout()
    elif type(target_layers) is list:
        for ind in target_layers:
            W0 = w[ind]
            if ind == target_layers[-1]:
                W0 = W0.transpose()
            fig = plt.figure()
            # mk_sz = 0.01 if W0.size>500 else 5; W = sparse.csr_matrix(W0); plt.spy(W,precision=tol,markersize=mk_sz)
            W = np.abs(W0)
            if binary_map:
                W[W < tol] = 0.0
                W[W > tol] = 1
            ax = sns.heatmap(
                W, xticklabels=[0, W.shape[1]], yticklabels=[0, W.shape[0]], cmap=cmap
            )
            if ind == 0:
                order = "$1^{st}$"
            elif ind == 1:
                order = "$2^{nd}$"
            elif ind == 2:
                order = "$3^{rd}$"
            else:
                order = "$%d^{th}$" % (ind + 1)

            if show_title:
                plt.title(
                    "%s The spasity of the %s layer of the network: %.4f"
                    % (ttl_tag, order, spasity[ind]),
                    loc="center",
                )
            plt.xticks([0, W0.shape[1]])
            plt.yticks([0, W0.shape[0]])
            plt.tight_layout()
            lst_w.append(W)
            lst_ax.append(ax)
            lst_order.append(order)
            if saveres:
                fig.savefig(
                    "{}{}spasity_layer{}".format(
                        outputPath, ttl_tag.rstrip(" ").replace(":", "_"), ind + 1
                    )
                )

    if concat_maps:
        fig, (ax, ax2) = plt.subplots(
            ncols=2, figsize=(8, 6), gridspec_kw={"width_ratios": [2, 1]}
        )
        fig.subplots_adjust(wspace=0.01)
        sns.heatmap(
            lst_w[-2],
            cmap=cmap,
            xticklabels=[1, lst_w[-2].shape[1]],
            yticklabels=[1, lst_w[-2].shape[0]],
            ax=ax,
            cbar=False,
        )
        fig.colorbar(
            ax.collections[0],
            ax=ax,
            location="left",
            use_gridspec=False,
            pad=0.1,
            ticks=[0.0, 0.1, 0.2, 0.3],
        )
        # ax.title.set_text('%s layer, spasity: %.4f'%(lst_order[-2],spasity[-2]))
        ax.set_yticks([0.5, lst_w[-2].shape[0] - 0.5])
        ax.set_xticks([0.5, lst_w[-2].shape[1] - 0.5])

        sns.heatmap(
            lst_w[-1],
            cmap=cmap,
            xticklabels=[1, lst_w[-1].shape[1]],
            yticklabels=[1, lst_w[-1].shape[0]],
            ax=ax2,
            cbar=False,
        )
        fig.colorbar(
            ax2.collections[0],
            ax=ax2,
            location="right",
            use_gridspec=False,
            pad=0.2,
            ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        )
        # ax2.title.set_text('%s layer, spasity: %.4f'%(lst_order[-1],spasity[-1]))
        ax2.yaxis.tick_right()
        ax2.set_yticks([0.5, lst_w[-1].shape[0] - 0.5])
        ax2.set_xticks([0.5, lst_w[-1].shape[1] - 0.5])
        ax2.tick_params(rotation=0)
        # fig.suptitle('{} Concatenation of the spasity graph for the last two layers'.format(ttl_tag), fontsize=14)
        if saveres:
            fig.savefig(
                "{}{}spasity_concat".format(
                    outputPath, ttl_tag.rstrip(" ").replace(":", "_")
                )
            )
        plt.show()


def visu_convLayer(model, target_index, save_fig=True, outDir="../results/"):
    """  
    Ít takes the model and target_index and visualize the chosen convolutional layer. 
        
    Return the spacity for the input matrix M                                   
    ----- INPUT                                                                 
        model            : (torch.nn.Module) the model itself
        target_index     : (int)  index to interested layers 
        save_fig         : (bool, optional) whether save the figure to local, default: True
        outDir           : (str, optional) the directory to save the figure, default: '../results/'
    """
    if target_index < 0:
        raise ValueError("Invalid value for 'target_index''")
    weights = {}
    i = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weights[i] = m.weight.data
            i += 1
    if target_index > i:
        raise ValueError("'target_index' exceeds the tolerant value")
    filters_ = weights[target_index]
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters_.min(), filters_.max()
    filters = (filters_ - f_min) / (f_max - f_min)
    n_filters, ix = filters.shape[0], 1
    lst_filters = []

    for i in range(n_filters):
        f = filters[i, :, :, :]
        l = []
        for j in range(filters.shape[1]):
            f_ = f[j, :, :]
            l.append(f_)
        lst_filters.append(l)

    C_out, C_in = filters_.size()[0], filters_.size()[1]
    if C_out > C_in:
        n_big, n_small = C_out, C_in
    else:
        n_big, n_small = C_in, C_out
    fig, axes = plt.subplots(
        n_small,
        n_big,
        figsize=(8, 4),
        sharex=True,
        sharey=False,
        gridspec_kw={"wspace": 0.2, "hspace": 0.05},
    )
    if n_small == 1:
        for i in range(n_big):
            axes[i].axis("off")
            axes[i].imshow(lst_filters[i][0], cmap="gray")
    else:
        for i in range(n_big):
            for j in range(n_small):
                axes[j, i].axis("off")
                axes[j, i].imshow(lst_filters[i][j], cmap="gray")
    return fig

    # -- End visu_convLayer --


def create_model_name(model, opt, VERSION=None):
    mdl_name = []
    mdl_name.append(model.__class__.__name__)
    mdl_name.append("_epoch-{}".format(N_EPOCHS))
    mdl_name.append("_mbatch-{}".format(BATCH_SIZE))
    if VERSION is not None:
        mdl_name.append("_version-{}".format(VERSION))
    mdl_name.append("_{}".format(opt.__class__.__name__))
    if "proximal" in opt.__class__.__name__.lower():
        mdl_name.append(
            "_gamma-{}".format(opt.state_dict()["param_groups"][0]["gamma"])
        )
        if "PGL1" in opt.__class__.__name__:
            ETA = opt.state_dict()["param_groups"][0]["eta"]
            mdl_name.append("_eta-{}".format(ETA))
        if "PGL21" in opt.__class__.__name__:
            mdl_name.append(
                "_eta-{}".format(opt.state_dict()["param_groups"][0]["eta"])
            )
            mdl_name.append(
                "_dim-{}".format(opt.state_dict()["param_groups"][0]["axis"])
            )
        if (
            "pgn" in opt.__class__.__name__.lower()
            or "nuclear" in opt.__class__.__name__.lower()
        ):
            mdl_name.append(
                "_etaStar-{}".format(opt.state_dict()["param_groups"][0]["eta_star"])
            )
        if "pglinf" in opt.__class__.__name__.lower():
            mdl_name.append(
                "etaInf-{}".format(opt.state_dict()["param_groups"][0]["eta"])
            )
        if opt.state_dict()["param_groups"][0]["control_sparsity"]:
            mdl_name.append("_CS")
    else:
        mdl_name.append("_lr-{}".format(opt.state_dict()["param_groups"][0]["lr"]))
    mdl_name = "".join(mdl_name)
    return mdl_name


def create_model_name_MultiOpts(model, lst_opt, VERSION=None):
    mdl_name = []
    mdl_name.append(model.__class__.__name__)
    mdl_name.append("_epoch-{}".format(N_EPOCHS))
    mdl_name.append("_mbatch-{}".format(BATCH_SIZE))
    if VERSION is not None:
        mdl_name.append("_version-{}".format(VERSION))
    for opt in lst_opt:
        if "proximal" in opt.__class__.__name__.lower():
            if "PGL1" in opt.__class__.__name__:
                mdl_name.append("_pgl1")
                mdl_name.append(
                    "-gamma-{}".format(opt.state_dict()["param_groups"][0]["gamma"])
                )
                ETA = opt.state_dict()["param_groups"][0]["eta"]
                mdl_name.append("-{}".format(ETA))
            if (
                "pgn" in opt.__class__.__name__.lower()
                or "nuclear" in opt.__class__.__name__.lower()
            ):
                mdl_name.append("_pgn")
                mdl_name.append(
                    "-gamma-{}".format(opt.state_dict()["param_groups"][0]["gamma"])
                )
                mdl_name.append(
                    "-{}".format(opt.state_dict()["param_groups"][0]["eta_star"])
                )
            if "pglinf" in opt.__class__.__name__.lower():
                mdl_name.append("_pglinf")
                mdl_name.append(
                    "-gamma-{}".format(opt.state_dict()["param_groups"][0]["gamma"])
                )
                mdl_name.append(
                    "-{}".format(opt.state_dict()["param_groups"][0]["eta"])
                )
            if opt.state_dict()["param_groups"][0]["control_sparsity"]:
                mdl_name.append("-CS")
        else:
            mdl_name.append("_{}".format(opt.__class__.__name__))
            mdl_name.append("_lr-{}".format(opt.state_dict()["param_groups"][0]["lr"]))
    mdl_name = "".join(mdl_name)
    return mdl_name


# In[9]:
# save and load net
# Two ways: save the net itself or just its paramesters
def save_net(model, filePath="../results/net.pkl"):
    """
    Save the whole model to the filePath.
    Note that the filePath contains also the name of file, 
    like \'../results/net.pkl\'
    """
    torch.save(model, filePath)


def save_net_params(model, filePath="../results/net_params.pkl"):
    """
    Save the parameters of the model to the filePath.
    Note that the filePath contains also the name of file, 
    like \'../results/net_params.pkl\'
    """
    torch.save(model.state_dict(), filePath)


def load_net(filePath="../results/net.pkl"):
    """
    Load the whole model to the filePath.
    Note that the filePath contains also the name of file,
    like \'../results/net.pkl\'
    """
    # return torch.load(filePath)
    return torch.load(filePath, map_location=lambda storage, loc: storage)


def load_net_params(model, filePath="../results/net_params.pkl"):
    """
    Load the parameters of the model to the filePath.
    Note that the filePath contains also the name of file, 
    like \'../results/net_params.pkl\'
    """
    model.load_state_dict(torch.load(filePath))
    # model.eval()
    return model


# =======================================================================
# =                     Running Scripts                                 =
# =======================================================================
# In[10]:
def run_loader(
    train_loader,
    test_loader,
    model,
    opt,
    loss_func,
    topk=5,
    version=None,
    doTest=True,
    showres=False,
    showprogress=True,
    keepfig=False,
    save_model=True,
    outputPath="../results/",
):
    """
    A running function with the train_loader and test_loader as input

    UPDATE:
    - It will now save the best model to local 
        (controlled by save_model whose default value is True).
        using the function save_net_params(model,outputPath+model_name+'.pkl')
        where model_name can be obtained by function create_model_name(model,opt)
    
    - It will now also calculate the top-K errors during the trainning.
    """
    keep_progressBar = True
    if keepfig == False:
        plt.close("all")
    check_cuda_model(model, USE_CUDA, N_EPOCHS)
    # Showing infos
    print("{:-^42}".format(""))
    print('Start Training for network "{}" with'.format(model._get_name()))
    print("{:<20}: {:<20}".format("Loss", loss_func._get_name()))
    print("{:<20}: {:<20}".format("Optimizer", opt.__class__.__name__))
    for group_dict in opt.state_dict()["param_groups"]:
        for (key, value) in group_dict.items():
            if "eta" in key:
                print("{:<23}({}:{})".format("", key, value))
            if "axis" in key:
                print("{:<23}({}:{})".format("", key, value))
    print("{:<20}: {:<20}".format("Batch_size", BATCH_SIZE))
    print("{:<20}: {:<20}".format("Test_batch_size", BATCH_SIZE))
    if version is not None:
        print("{:<20}: {:<20}".format("Version", version))
    print("{:-^42}".format(""))
    # Training
    perfs = []
    lst_train_loss = []
    lst_test_loss = []
    lst_acc = []
    topk_err = []
    best_score = 0

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        if showprogress:
            model, train_loss = train(
                epoch,
                train_loader,
                model,
                opt,
                loss_func,
                keep_pbar=keep_progressBar,
                verbose=False,
            )
        else:
            model, train_loss = train_novisu(
                epoch, train_loader, model, opt, loss_func, verbose=False
            )
        # do the test part
        if doTest:
            test_loss, correct = test(test_loader, model, loss_func, verbose=False)
            # Weight matrix sparsity
            weights, spasity_w = weights_and_sparsity(model)
            spasity = list(spasity_w.values())
            # neuron sparsity
            _, sps_neuron = weights_and_sparsityByAxis(model, tol=1.0e-3, axis=0)
            sps_neuron = list(sps_neuron.values())
            # TopK
            TE = topk_error(topk, test_loader, model, verbose=False)
            topk_error_rate = float(TE[0] / TE[1])
            topk_err.append(topk_error_rate)
            # Performances
            perfs.append(
                [
                    epoch,
                    train_loss,
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    topk_error_rate,
                    time.perf_counter() - t0,
                ]
                + spasity
                + sps_neuron
            )
            # Test loss
            lst_test_loss.append(test_loss)
            # Save the best model
            if correct > best_score:
                model_name = create_model_name(model, opt, VERSION=version)
                save_net_params(model, outputPath + model_name + ".pkl")
                best_score = correct
            if showres:
                print(
                    "\nEpoch {}: train loss {:.4f}, test loss {:.4f}, best accuracy {:.4f}, accuracy {}/{}, top{} error {}, in {:.2f}s\n".format(
                        perfs[-1][0],
                        perfs[-1][1],
                        perfs[-1][2],
                        best_score / perfs[-1][4],
                        perfs[-1][3],
                        perfs[-1][4],
                        topk,
                        perfs[-1][5],
                        perfs[-1][6],
                    )
                )
        else:
            perfs.append([epoch, train_loss, time.perf_counter() - t0])
        lst_train_loss.append(train_loss)
        lst_acc.append(correct / len(test_loader.dataset))

    if not doTest:
        print("Saving model while not performing testing")
        model_name = create_model_name(model, opt, VERSION=version)
        save_net_params(model, outputPath + model_name + ".pkl")
    # processing perds as table
    perfs = np.array(perfs)
    perfs[:, 1] = perfs[:, 1] / perfs[0, 1]
    perfs[:, 2] = perfs[:, 2] / perfs[0, 2]
    df_perfs = pd.DataFrame(
        perfs,
        columns=[
            "Epoch",
            "Train loss",
            "Test loss",
            "Correct nummber",
            "Total number(n_test)",
            "Top%d Error" % topk,
            "Time elapsed",
        ]
        + ["sparsity layer {}".format(il + 1) for il in range(len(spasity))]
        + ["neuron sparsity layer{}".format(il + 1) for il in range(len(sps_neuron))],
    )
    # Showing result
    if showres:
        fig_lossEpoch = plt.figure(figsize=(8, 6))
        plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, perfs[:, 1], label="Train loss")
        if doTest:
            plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, perfs[:, 2], label="Test loss")
        plt.title(
            "loss for each epoch {}\n(Loss function:{})".format(
                opt.__class__.__name__, loss_func._get_name()
            ),
            fontsize=18,
        )
        plt.ylabel("Loss", fontsize=16)
        plt.xlabel("Epoch", fontsize=16)
        plt.xticks(
            np.linspace(1, N_EPOCHS, num=min(N_EPOCHS, 6), endpoint=True, dtype=int)
        )
        plt.xlim(left=1, right=N_EPOCHS)
        plt.legend()

        if doTest:
            fig_accEpoch = plt.figure(figsize=(8, 6))
            plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, lst_acc)
            plt.title(
                "Accuracy for each epoch {}\n(Loss function:{})".format(
                    opt.__class__.__name__, loss_func._get_name()
                ),
                fontsize=18,
            )
            plt.ylabel("Accuracy", fontsize=16)
            plt.xlabel("Epoch", fontsize=16)
            plt.xticks(
                np.linspace(1, N_EPOCHS, num=min(N_EPOCHS, 6), endpoint=True, dtype=int)
            )
            plt.xlim(left=1, right=N_EPOCHS)
            plt.ylim(top=1.0)
    return model, df_perfs


def run_loader_multiOpts(
    train_loader,
    test_loader,
    model,
    opts,
    loss_func,
    topk=5,
    version=None,
    doTest=True,
    showres=False,
    showprogress=True,
    keepfig=False,
    save_model=True,
    outputPath="../results/",
):
    """
    A running function with the train_loader and test_loader as input
    
    Case multiple optimizers are given
    """
    from matplotlib import pyplot as plt

    if type(opts) is not list:
        raise ValueError("Wrong type of input :{}".format(opts))
    keep_progressBar = True
    if keepfig == False:
        plt.close("all")
    check_cuda_model(model, USE_CUDA, N_EPOCHS)
    # Showing infos
    print("{:-^42}".format(""))
    print('Start Training for network "{}" with'.format(model._get_name()))
    print("{:<20}: {:<20}".format("Loss", loss_func._get_name()))

    print(
        "".join(
            [" {:<20}: \n".format("Optimizers")]
            + [
                " {:<20}  {:<20}\n".format(
                    "",
                    opt.__class__.__name__
                    + (
                        "(without constraint)"
                        if opt.state_dict()["param_groups"][0]["control_sparsity"]
                        else ""
                    ),
                )
                for opt in opts
            ]
        )
    )
    print("{:<20}: {:<20}".format("Batch_size", BATCH_SIZE))
    print("{:<20}: {:<20}".format("Test_batch_size", BATCH_SIZE))
    if version is not None:
        print("{:<20}: {:<20}".format("Version", version))
    print("{:-^42}".format(""))

    # Training
    perfs = []
    lst_train_loss = []
    lst_test_loss = []
    lst_acc = []
    log = []
    topk_err = []
    best_score = 0

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        if showprogress:
            model, train_loss = train_multiOpts(
                epoch,
                train_loader,
                model,
                opts,
                loss_func,
                keep_pbar=keep_progressBar,
                verbose=False,
            )
        else:
            model, train_loss = train_novisu_multiOpts(
                epoch, train_loader, model, opts, loss_func, verbose=False, log=log
            )
        # do the test part
        if doTest:
            test_loss, correct = test(test_loader, model, loss_func, verbose=False)
            # Weight matrix sparsity
            weights, spasity_w = weights_and_sparsity(model)
            spasity = list(spasity_w.values())
            # neuron sparsity
            _, sps_neuron = weights_and_sparsityByAxis(model, tol=1.0e-3, axis=0)
            sps_neuron = list(sps_neuron.values())
            # TopK
            TE = topk_error(topk, test_loader, model, verbose=False)
            topk_error_rate = float(TE[0] / TE[1])
            topk_err.append(topk_error_rate)
            # Performances
            perfs.append(
                [
                    epoch,
                    train_loss,
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    topk_error_rate,
                    time.perf_counter() - t0,
                ]
                + spasity
                + sps_neuron
            )
            lst_test_loss.append(test_loss)

            # Save the best model
            if correct > best_score:
                model_name = create_model_name_MultiOpts(model, opts, VERSION=version)
                save_net_params(model, outputPath + model_name + ".pkl")
                best_score = correct
            if showres:
                print(
                    "\nEpoch {}: train loss {:.4f}, test loss {:.4f}, best accuracy {:.4f}, accuracy {}/{}, top{} error {}, in {:.2f}s\n".format(
                        perfs[-1][0],
                        perfs[-1][1],
                        perfs[-1][2],
                        best_score / perfs[-1][4],
                        perfs[-1][3],
                        perfs[-1][-4],
                        topk,
                        perfs[-1][5],
                        perfs[-1][6],
                    )
                )
        else:
            perfs.append([epoch, train_loss, time.perf_counter() - t0])
        lst_train_loss.append(train_loss)
        lst_acc.append(correct / len(test_loader.dataset))
    # processing perds as table
    perfs = np.array(perfs)
    perfs[:, 1] = perfs[:, 1] / perfs[0, 1]
    perfs[:, 2] = perfs[:, 2] / perfs[0, 2]
    df_perfs = pd.DataFrame(
        perfs,
        columns=[
            "Epoch",
            "Train loss",
            "Test loss",
            "Correct nummber",
            "Total number(n_test)",
            "Top%d Error" % topk,
            "Time elapsed",
        ]
        + ["sparsity layer {}".format(il + 1) for il in range(len(spasity))]
        + ["neuron sparsity layer{}".format(il + 1) for il in range(len(sps_neuron))],
    )
    # Showing result
    if showres:
        fig_lossEpoch = plt.figure(figsize=(8, 6))
        plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, perfs[:, 1], label="Train loss")
        if test:
            plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, perfs[:, 2], label="Test loss")
        plt.title(
            "loss for each epoch \n(Loss function:{})".format(loss_func._get_name()),
            fontsize=18,
        )
        plt.ylabel("Loss", fontsize=16)
        plt.xlabel("Epoch", fontsize=16)
        plt.xticks(
            np.linspace(1, N_EPOCHS, num=min(N_EPOCHS, 6), endpoint=True, dtype=int)
        )
        plt.xlim(left=1, right=N_EPOCHS)
        plt.legend()

        if test:
            fig_accEpoch = plt.figure(figsize=(8, 6))
            plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, lst_acc)
            plt.title(
                "Accuracy for each epoch \n(Loss function:{})".format(
                    loss_func._get_name()
                ),
                fontsize=18,
            )
            plt.ylabel("Accuracy", fontsize=16)
            plt.xlabel("Epoch", fontsize=16)
            plt.xticks(
                np.linspace(1, N_EPOCHS, num=min(N_EPOCHS, 6), endpoint=True, dtype=int)
            )
            plt.xlim(left=1, right=N_EPOCHS)
            plt.ylim(top=1.0)
    return model, df_perfs


def run(
    x,
    y,
    nb_clusters,
    model,
    opt,
    loss_func,
    validation_split=0.25,
    topk=5,
    Dotest=True,
    random_seed=None,
    showres=False,
    showprogress=True,
    keepfig=False,
):
    """
    A running function with the orginal data as input, we regards x as 
    the dataset which is not yet been divided into training set and validation set
    so the function will perfrom first a random sampling with validation_split as 
    the spliting rate for the validation set.
    """
    keep_progressBar = True
    if keepfig == False:
        plt.close("all")
    check_cuda_model(model, USE_CUDA, N_EPOCHS)
    if USE_CUDA:
        device = "cuda"
    else:
        device = None

    x = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)
    y = torch.as_tensor(y, device=device)
    x.requires_grad_()
    y.requires_grad_()
    if y.unique(sorted=True)[0] == 1:
        # Note that this modification is applied directly on the y and
        # what lies in y are exactly the values from the original y
        # so the modification will be applied on the orginal y.
        # That is to say, the original y will be changed (y=y-1)
        y.add_(-1)
        print("Found inpropriate labels. Labels are remodified.")

    print(
        "Start Training for network with\n {:<11}: {:<20}\n {:<11}: {:<20}".format(
            "Loss", loss_func._get_name(), "Optimizer", opt.__class__.__name__
        )
    )

    # Creating data indices for training and validation splits:
    dataset_size = x.shape[0]
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)

    torch_dataset = Data.TensorDataset(x, y)  # put dateset into torch dataset
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        sampler=train_sampler,
    )

    test_loader = Data.DataLoader(
        torch_dataset, batch_size=BATCH_SIZE, num_workers=2, sampler=test_sampler
    )
    # Training
    perfs = []
    lst_train_loss = []
    lst_test_loss = []
    lst_acc = []
    topk_err = []
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        # Train
        if showprogress:
            model, train_loss = train(
                epoch,
                train_loader,
                model,
                opt,
                loss_func,
                keep_pbar=keep_progressBar,
                verbose=False,
            )
        else:
            model, train_loss = train_novisu(
                epoch, train_loader, model, opt, loss_func, verbose=False
            )
        # Test if choose to
        if Dotest:
            test_loss, correct = test(test_loader, model, loss_func, verbose=False)
            # Weight matrix sparsity
            weights, spasity_w = weights_and_sparsity(model)
            spasity = list(spasity_w.values())
            # neuron sparsity (For now we only have linear layers)
            _, sps_neuron = weights_and_sparsityByAxis(model, tol=1.0e-3, axis=0)
            sps_neuron = list(sps_neuron.values())
            # TopK
            TE = topk_error(topk, test_loader, model)
            topk_error_rate = float(TE[0] / TE[1])
            topk_err.append(topk_error_rate)
            # Performances
            perfs.append(
                [
                    epoch,
                    train_loss,
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    topk_error_rate,
                    time.perf_counter() - t0,
                ]
                + spasity
                + sps_neuron
            )
            lst_test_loss.append(test_loss)
            if showres:
                print(
                    "\nEpoch {}: train loss {:.4f}, test loss {:.4f}, accuracy {}/{}, top{} error {}, in {:.2f}s\n".format(
                        perfs[-1][0],
                        perfs[-1][1],
                        perfs[-1][2],
                        perfs[-1][3],
                        perfs[-1][4],
                        topk,
                        perfs[-1][5],
                        perfs[-1][6],
                    )
                )
        else:
            perfs.append([epoch, train_loss, time.perf_counter() - t0])
        lst_train_loss.append(train_loss)
        lst_test_loss.append(test_loss)
        lst_acc.append(correct / len(test_loader.dataset))
    # processing perds as table
    perfs = np.array(perfs)
    # Showing result
    if showres:
        fig_lossEpoch = plt.figure(figsize=(8, 6))
        plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, perfs[:, 1], label="Train loss")
        plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, perfs[:, 2], label="Test loss")
        plt.title(
            "loss for each epoch {}\n(Loss function:{})".format(
                opt.__class__.__name__, loss_func._get_name()
            ),
            fontsize=18,
        )
        plt.ylabel("Loss", fontsize=16)
        plt.xlabel("Epoch", fontsize=16)
        plt.xticks(
            np.linspace(1, N_EPOCHS, num=min(N_EPOCHS, 6), endpoint=True, dtype=int)
        )
        plt.xlim(left=1, right=N_EPOCHS)
        plt.legend()

        fig_accEpoch = plt.figure(figsize=(8, 6))
        plt.plot(np.arange(N_EPOCHS, dtype=int) + 1, lst_acc)
        plt.title(
            "Accuracy for each epoch {}\n(Loss function:{})".format(
                opt.__class__.__name__, loss_func._get_name()
            ),
            fontsize=18,
        )
        plt.ylabel("Accuracy", fontsize=16)
        plt.xlabel("Epoch", fontsize=16)
        plt.xticks(
            np.linspace(1, N_EPOCHS, num=min(N_EPOCHS, 6), endpoint=True, dtype=int)
        )
        plt.xlim(left=1, right=N_EPOCHS)
        plt.ylim(top=1.0)
    perfs[:, 1] = perfs[:, 1] / perfs[0, 1]
    perfs[:, 2] = perfs[:, 2] / perfs[0, 2]
    df_perfs = pd.DataFrame(
        perfs,
        columns=[
            "Epoch",
            "Train loss",
            "Test loss",
            "Correct nummber",
            "Total number(n_test)",
            "Top%d Error" % topk,
            "Time elapsed",
        ]
        + ["sparsity layer {}".format(il + 1) for il in range(len(spasity))]
        + ["neuron sparsity layer{}".format(il + 1) for il in range(len(sps_neuron))],
    )
    return model, df_perfs


# ====== Top-k error
def topk_error(k, test_loader, model, verbose=True):

    """
    Determination of top k error rate
    
    param:
        - k : int>0
        - test_loader
        - model
        - verbose : True/False
        
    returns:
        [n1,total]:
            - n1 : int - Number of instance fulfilling top-k
            - total: int - total number of instances
            
    """

    model.eval()
    test_loss = 0
    correct = 0

    topk = 0
    total = 0
    # q=0
    for data, target in test_loader:

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data.requires_grad_()

        output = model(data.type(dtype=torch.get_default_dtype()))

        target_len = target.size()[0]

        new_target = target.reshape(1, target_len).reshape(target_len, 1)

        # Z0 = torch.add(-1*new_target,torch.topk(output,k).indices)
        """From the doc of torch.topk:
        A tuple of (values, indices) is returned, 
        where the indices are the indices of the elements in the original input tensor.
        So the indices here will be the second one of the output
        """
        Z0 = torch.add(-1 * new_target, torch.topk(output, k)[1])
        Z = Z0.data.cpu().numpy()

        local_count = 0

        for j in range(Z.shape[0]):
            if 0 in Z[j]:
                local_count = local_count + 1

        topk += local_count
        total += Z.shape[0]

    if verbose:
        print("Test top-" + str(k) + " :" + str(topk) + " on " + str(total))

    return [topk, total]

