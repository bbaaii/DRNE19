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

    #loss = 0
    #loss = loss.type(torch.cuda.FloatTensor)
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

    def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha,normal_loss,seed,device,use_point_stn=True, use_feat_stn=True, use_mask=True):
        '''
        Constructor.

        hyps -- number of line hypotheses sampled for each image
        inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
        inlier_beta -- scaling factor within the sigmoid of the soft inlier count
        inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
        loss_function -- function to compute the quality of estimated line parameters wrt ground truth
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
        #self.dsacpnet = DSACPNet( num_points=500)
        self.pcpnet = PCPNet(use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, use_mask=use_mask)
        self.device=device


    def __sample_hyp(self,pts,hyps):
        '''
        Calculate a line hypothesis (slope, intercept) from two random points.

        x -- vector of x values
        y -- vector of y values
        z -- vector of z values
        '''
        # select three random points
        #print(x,y,z)
        batchsize = pts.size(0)
        
        
        plane_num = hyps
        while 1:
            
            #plane_num = int(pts.size(0)/6+1)#平面数，即此处处理的batchsize
            
            index=torch.stack([torch.from_numpy(np.stack(self.rng.choice(pts.size(1), 3, replace=False) for _ in range(hyps)).reshape(-1)) for _ in range(batchsize)]).to(self.device)#.long()
            # plane_num*9
            index = index.view(batchsize,hyps*3,1)
            nindex=torch.cat((index,index,index),2)
            # nindex=index.new_empty(batchsize,hyps*3,3).cuda(0)
            # nindex[:,:,0]=index
            # nindex[:,:,1]=index
            # nindex[:,:,2]=index
            pts_sample=torch.gather(pts, 1, nindex)
            pts_sample = pts_sample.view(pts.size(0),plane_num,3,3)# plane_num*3*3
            pts_sample =pts_sample.transpose(3,2)
            planes_sample = utils.pts_to_plane(pts_sample,plane_num)
            # print("选择的点")
            # print(pts_sample)
            nozeros= 4-torch.eq(planes_sample,0).sum(2)

            if (len(torch.nonzero(nozeros)) == (batchsize*plane_num)) : break
            else:print("fail")
        #else: print("fail")
        # print("扣掉无效的平面")
        # print(planes_sample)
        
        
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

        prediction -- predicted 2D points for a batch of images, array of shape (Bx2) where
            B is the number of images in the batch
            2 is the number of point dimensions (y, x)
        labels -- ground truth labels for the batch, array of shape (Bx2) where
            B is the number of images in the batch
            2 is the number of parameters (intercept, slope)
        '''
        
        pts,trans,_,mask=self.pcpnet(x)
        if self.use_point_stn:
            pts=torch.bmm(pts,trans.transpose(2, 1))
        batchsize=pts.size(0)

        planes = self.__sample_hyp(pts,self.hyps)

                # === step 2: score hypothesis using soft inlier count ====

        score  = self.__soft_inlier_count(pts,planes)
                #print(score)
                #print(inliers)
                # === step 3: refine hypothesis ===========================

        #norplanes=[]#planes.new_empty(batchsize,32,3)
        norplanes0=planes[:,:,0]/torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2)
        norplanes1=planes[:,:,1]/torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2)
        norplanes2=planes[:,:,2]/torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2)
        norplanes = torch.cat(( norplanes0.view(batchsize,self.hyps,1) , norplanes1.view(batchsize,self.hyps,1) , norplanes2.view(batchsize,self.hyps,1)),2)
        
        loss = compute_loss(
            #torch.bmm(norplanes,trans.transpose(2, 1)),
            norplanes,
            target.view(batchsize,1,3),
                    #output_loss_weight =output_loss_weight,
            normal_loss = self.normal_loss
                    
        )
        
        maxindex = torch.argmax(score,1).long().view(batchsize,1)
        top_loss=torch.gather(loss, 1, maxindex)#.mean()

        maxindex = maxindex.view(batchsize,1,1)
        nindex=torch.cat((maxindex,maxindex,maxindex),2)
        # nindex=maxindex.new_empty(batchsize,1,3).cuda(0)
        # nindex[:,:,0]=maxindex
        # nindex[:,:,1]=maxindex
        # nindex[:,:,2]=maxindex

        # === step 5: calculate the expectation ===========================

        #softmax distribution from hypotheses scores            
        score = F.softmax(self.inlier_alpha * score, 1)
        
        exp_loss = torch.sum(loss * score,1).mean()


    
        return exp_loss, top_loss,torch.gather(norplanes, 1, nindex).squeeze(),pts,mask

