from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
import numpy as np

from pointnet2.utils import pointnet2_utils

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        #new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )
        
        for i in range(len(self.groupers)):
            grouped_xyz = self.groupers[i](
                xyz, new_xyz, None
            )  # (B, C, npoint, nsample)
            
            # print(i,grouped_xyz.size())
            # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            # new_features = F.max_pool2d(
            #     new_features, kernel_size=[1, new_features.size(3)]
            # )  # (B, mlp[-1], npoint, 1)
            # new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            # new_features_list.append(new_features)

        return grouped_xyz#torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples,  bn=True, use_xyz=True):#mlps,
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) #== len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        #self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            #mlp_spec = mlps[i]
            #if use_xyz:
            #    mlp_spec[0] += 3

            #self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1568546)
    torch.cuda.manual_seed_all(1698416)
    # xyz = torch.empty(50,3)
    # rng = np.random.RandomState(2)
    # xyz[:,0] = torch.tensor(rng.choice(50, 50, replace=False))
    # xyz[:,1] = torch.tensor(rng.choice(50, 50, replace=False))
    # for i in range(50):
    #     xyz[i][2] = 1-xyz[i][0]-xyz[i][1]
    xyz =torch.tensor([[ 1.2474, 14.3901, -2.5374],
        [ 1.2527, 14.3823, -2.5360],
        [ 1.2769, 14.3919, -2.5377],
        [ 1.2779, 14.3881, -2.5370],
        [ 1.2925, 14.3844, -2.5364],
        [ 1.2412, 14.4598, -2.5496],
        [ 1.2485, 14.4535, -2.5485],
        [ 1.2494, 14.4046, -2.5399],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0]], device='cuda:0').view(1,-1,3)
    #xyz = Variable(torch.randn( 2,100, 3).cuda(), requires_grad=True)
    # print("xyz")
    # print(xyz)
    # xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)
    # print("xyz_feats")
    # print(xyz_feats)
    test_module = PointnetSAModuleMSG(
        npoint=32, radii=[0.05], nsamples=[16]#, mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    #print("test_module(xyz, xyz_feats)")
    #print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        grouped_xyz = test_module(xyz)
        #grouped_xyz.backward(torch.cuda.FloatTensor(*grouped_xyz.size()).fill_(1))
        # print("new_features")
        # print(new_features)
        # print("xyz.grad")
        # print(xyz.grad)
