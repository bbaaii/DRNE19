import torch
import torch.nn.functional as F
import numpy as np
import random
import utils
import math
import torch
import torch.nn as nn
import torchsnooper

from pcpnet import PCPNet
import os
from visdom import Visdom
def compute_loss(pred, target,  normal_loss):

    if normal_loss == 'ms_euclidean':
        loss = torch.min((pred-target).pow(2).sum(2), (pred+target).pow(2).sum(2))#* output_loss_weight
    elif normal_loss == 'ms_oneminuscos':
        loss = (1-torch.abs(utils.cos_angle(pred, target))).pow(2)#* output_loss_weight
    else:
        raise ValueError('Unsupported loss type: %s' % (normal_loss))

    return loss


class DSAC(nn.Module):
    '''
    Differentiable RANSAC to robustly fit planes.
    '''

    def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha,normal_loss,seed,device,use_point_stn=True, use_feat_stn=True, use_mask=True,points_num=32,points_per_patch=512,sym_op='max'):
        '''
        Constructor.

        hyps -- number of planes hypotheses sampled for each patch
        inlier_thresh -- threshold used in the soft inlier count, 
        inlier_beta -- scaling factor within the sigmoid of the soft inlier count
        inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)

        '''
        super(DSAC, self).__init__()
        self.hyps = hyps
        self.inlier_thresh = inlier_thresh
        self.inlier_beta = inlier_beta
        self.inlier_alpha = inlier_alpha
        self.normal_loss = normal_loss
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.use_point_stn=use_point_stn
        self.use_feat_stn=use_feat_stn
        self.use_mask=use_mask
        self.pcpnet = PCPNet(num_points=points_per_patch,output_dim=3*points_num,use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, use_mask=use_mask,sym_op=sym_op)
        self.device=device


    def __sample_hyp(self,pts,hyps):
        '''
        Calculate a plane hypothesis  from 3 random points.

        '''
        batchsize = pts.size(0)
        plane_num = hyps
        while 1:
            # select three*plane_num random points
            index=torch.stack([torch.from_numpy(np.stack(self.rng.choice(pts.size(1), 3, replace=False) for _ in range(hyps)).reshape(-1)) for _ in range(batchsize)]).to(self.device)#.long()

            index = index.view(batchsize,hyps*3,1)
            nindex=torch.cat((index,index,index),2)

            pts_sample=torch.gather(pts, 1, nindex)
            pts_sample = pts_sample.view(pts.size(0),plane_num,3,3)# plane_num*3*3
            pts_sample =pts_sample.transpose(3,2)
            planes_sample = utils.pts_to_plane(pts_sample,plane_num)

            nozeros= 4-torch.eq(planes_sample,0).sum(2)

            if (len(torch.nonzero(nozeros)) == (batchsize*plane_num)) : break

        return planes_sample  # True indicates success

    def __soft_inlier_count(self,pts,planes):
        '''
        Soft inlier count for a given plane and a given set of points.

        hs -- Four parameters of the plane
        x -- vector of x values
        y -- vector of y values
        z -- vector of z values
        '''

        # point plane distances
        batchsize=pts.size(0)

        dists = torch.abs(planes[:,:,0].view(batchsize,-1,1)*pts[:,:,0].view(batchsize,1,-1)+planes[:,:,1].view(batchsize,-1,1)*pts[:,:,1].view(batchsize,1,-1)+planes[:,:,2].view(batchsize,-1,1)*pts[:,:,2].view(batchsize,1,-1)+planes[:,:,3].view(batchsize,-1,1))#.cuda()

        dists = dists / torch.sqrt(planes[:,:,0].view(batchsize,-1,1)**2+planes[:,:,1].view(batchsize,-1,1)**2+planes[:,:,2].view(batchsize,-1,1)**2)#.cuda()
        dists = 1- torch.sigmoid(self.inlier_beta * (dists- self.inlier_thresh) )

        score = torch.sum(dists,2)#.cuda()

        return score#, dists


    def forward(self,x, target):
        '''
        Perform robust, differentiable plane fitting according to DSAC.

        Returns the expected loss of choosing a good plane hypothesis which can be used for backprob.

        '''
        
        pts,trans,_,mask=self.pcpnet(x)
        if self.use_point_stn:
            pts=torch.bmm(pts,trans.transpose(2, 1))
        batchsize=pts.size(0)
        # === step 1: select  planes ===========================
        planes = self.__sample_hyp(pts,self.hyps)

        # === step 2: score hypothesis using soft inlier count ====

        score  = self.__soft_inlier_count(pts,planes)

        norplanes0=planes[:,:,0]/torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2)
        norplanes1=planes[:,:,1]/torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2)
        norplanes2=planes[:,:,2]/torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2)
        norplanes = torch.cat(( norplanes0.view(batchsize,self.hyps,1) , norplanes1.view(batchsize,self.hyps,1) , norplanes2.view(batchsize,self.hyps,1)),2)
        # === step 3: calculate the loss ===========================
        loss = compute_loss(
            norplanes,
            target.view(batchsize,1,3),
            normal_loss = self.normal_loss
        )
        
        maxindex = torch.argmax(score,1).long().view(batchsize,1)
        top_loss=torch.gather(loss, 1, maxindex)#.mean()

        maxindex = maxindex.view(batchsize,1,1)
        nindex=torch.cat((maxindex,maxindex,maxindex),2)
        

        # === step 4: calculate the expectation ===========================

        #softmax distribution from hypotheses scores            
        score = F.softmax(self.inlier_alpha * score, 1)
        
        exp_loss = torch.sum(loss * score,1).mean()

        return exp_loss, top_loss,torch.gather(norplanes, 1, nindex).squeeze(),pts,mask

