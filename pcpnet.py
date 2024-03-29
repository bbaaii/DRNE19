from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import utils


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=512, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x
class MASK(nn.Module):
    def __init__(self, num_points=512, dim=3 ):
        super(MASK, self).__init__()
        self.dim = dim
            #self.num_scales = num_scales
    
        self.num_points = num_points
    
        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.conva = torch.nn.Conv1d(576, 256, 1)
        #self.convb = torch.nn.Conv1d(512, 256, 1)
        self.convc = torch.nn.Conv1d(256, 64, 1)
        self.convd = torch.nn.Conv1d(64, 1, 1)
        self.bna = nn.BatchNorm1d(256)
        #self.bnb = nn.BatchNorm1d(256)
        self.bnc = nn.BatchNorm1d(64)
    
        #self.fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512,num_points)
            #self.fc3 = nn.Linear(256, 4)
    
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        #self.bn4 = nn.BatchNorm1d(512)
            #self.bn5 = nn.BatchNorm1d(256)
    
    
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))#3 to 64
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))# 64 to 128
    
        x = F.relu(self.bn3(self.conv3(x)))# 128 to 512
    
        x = self.mp1(x)
    
        x = x.view(-1, 512)
        x = x.view(-1, 512, 1).repeat(1, 1, n_pts)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bna(self.conva(x)))
        #x = F.relu(self.bnb(self.convb(x)))
        x = F.relu(self.bnc(self.convc(x)))
        x = F.relu(self.convd(x))

        #x = F.relu(self.bn4(self.fc1(x)))
    
            #x = F.relu(self.bn5(self.fc2(x)))
    
        #x = F.relu(self.fc2(x))
        x=x.view(batchsize,1,self.num_points)
        return x#,x2
class QSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=512, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = utils.batch_quat_to_rotmat(x)

        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=512, use_point_stn=True, use_mask=True,use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(PointNetfeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_mask=use_mask
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op)
        if self.use_mask:
            self.conva = torch.nn.Conv1d(1088, 256, 1)
            #self.convb = torch.nn.Conv1d(512, 256, 1)
            self.convc = torch.nn.Conv1d(256, 64, 1)
            self.convd = torch.nn.Conv1d(64, 1, 1)
            self.bna = nn.BatchNorm1d(256)
            #self.bnb = nn.BatchNorm1d(256)
            self.bnc = nn.BatchNorm1d(64)
        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)
        # if self.use_mask:
        #     self.conv0a = torch.nn.Conv1d(4*self.point_tuple, 64, 1)
        # else:
        self.conv0a = torch.nn.Conv1d(3*self.point_tuple, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(1024, 1024*self.num_scales, 1)
            self.bn4 = nn.BatchNorm1d(1024*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)
        else:
            trans = None
        
        # if self.use_mask:
        #     mask = self.mask(x)
            
        #     x = torch.cat([x, mask], 1)
        # else:
        #     mask=None
            
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))
        
            
        # mlp (64,64)
        

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None
        
        
            
        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_mask:
            pointfeat = x
            n_pts = x.size()[2]
        x = F.relu(self.bn2(self.conv2(x)))
        
        
        x = self.bn3(self.conv3(x))
        
        # mlp (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(F.relu(x)))

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None # so the intermediate result can be forgotten if it is not needed

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales**2, 1)
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales
        #x= x.transpose(2,1)
        x = x.contiguous().view(-1, 1024*self.num_scales**2)
        if self.use_mask:
            x2 = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            x2 = torch.cat([x2, pointfeat], 1)
            x2 = F.relu(self.bna(self.conva(x2)))
            #x2 = F.relu(self.bnb(self.convb(x2)))
            x2 = F.relu(self.bnc(self.convc(x2)))
            mask = self.convd(x2)
        else:mask = None
        return x, trans, trans2, mask


class PCPNet(nn.Module):
    def __init__(self, num_points=512, output_dim=3*32, use_point_stn=True, use_feat_stn=True, use_mask=True,sym_op='max', get_pointfvals=False, point_tuple=1):
        super(PCPNet, self).__init__()
        self.num_points = num_points

        self.feat = PointNetfeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple,
            use_mask=use_mask)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3)
    def forward(self, x):
        x, trans, trans2, mask = self.feat(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)
        x = x.contiguous().view(-1,32,3)
        #x= x.transpose(2,1)
        x = torch.sigmoid(x)*2-1.0
        return x, trans, trans2, mask

class MSPCPNet(nn.Module):
    def __init__(self, num_scales=2, num_points=512, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(MSPCPNet, self).__init__()
        self.num_points = num_points

        self.feat = PointNetfeat(
            num_points=num_points,
            num_scales=num_scales,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)
        self.fc0 = nn.Linear(1024*num_scales**2, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3)
    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)

        return x, trans, trans2, pointfvals
