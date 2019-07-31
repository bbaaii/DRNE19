import torch
import torch.nn.functional as F
import numpy as np
import random
import utils
import math
import torch
import torch.nn as nn
import torchsnooper
from dsacpnet import  DSACPNet
from pcpnet import PCPNet
import os
def gaussian(x):
    sigma = torch.sqrt(((x-x.mean())**2).mean())
    
    gaussian_kernel = (1./(2.*math.pi*sigma)) *\
                  torch.exp(
                      -((x -x.mean())**2.) /\
                      (2.*sigma**2.)
                  )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel
def compute_loss(pred, target,  normal_loss):

    #loss = 0
    #loss = loss.type(torch.cuda.FloatTensor)
    if normal_loss == 'ms_euclidean':
        loss = torch.min((pred-target).pow(2).sum(1), (pred+target).pow(2).sum(1))#* output_loss_weight
    elif normal_loss == 'ms_oneminuscos':
        loss = (1-torch.abs(utils.cos_angle(pred, target))).pow(2)#* output_loss_weight
    else:
        raise ValueError('Unsupported loss type: %s' % (normal_loss))

    return loss

def iszero(hs):
    i=1
    #for h in hs:
    if hs!=0 and not (math.isnan(hs)):
        i=0
            #break
    return i

class DSAC(nn.Module):
    '''
    Differentiable RANSAC to robustly fit planes.
    '''

    def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha,normal_loss,seed):
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
        #self.dsacpnet = DSACPNet( num_points=500)
        self.pcpnet = PCPNet()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc0 = nn.Linear(512, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64,3)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        #self.loss_function = loss_function

    def __sample_hyp(self,pts,hyps):
        '''
        Calculate a line hypothesis (slope, intercept) from two random points.

        x -- vector of x values
        y -- vector of y values
        z -- vector of z values
        '''
        # select three random points
        #print(x,y,z)
        flag = 1
        tries = 10
        plane_num = hyps
        while flag:
            tries = tries-1
            #plane_num = int(pts.size(0)/6+1)#平面数，即此处处理的batchsize
            
            pts_sample = pts[np.stack(self.rng.choice(pts.size(0), 3, replace=False) for _ in range(hyps)).reshape(-1)]# plane_num*9
            
            pts_sample = pts_sample.view(plane_num,3,3)# plane_num*3*3
            pts_sample =pts_sample.transpose(2,1)
            # print("选择的点")
            # print(pts_sample)
            planes_sample = utils.pts_to_plane(pts_sample)#  plane_num×4
            # print("计算出的平面")
            # print(planes_sample)
            index = torch.nonzero(planes_sample)
            if index.size(0)%4==0:
                index = index[:,0].view(-1,4)[:,0]
            else:
                offset = torch.zeros(4-index.size(0)%4,2).long().cuda()
                index=torch.cat((index,offset),0)
                index = index[:,0].view(-1,4)[:,0]
            planes_sample=planes_sample[index,:]
            if (planes_sample.size(0) > 0) or (tries < 0) : flag = 0
            #else: print("fail")
        # print("扣掉无效的平面")
        # print(planes_sample)
        if tries < 0: return None,False
        
        return planes_sample,True  # True indicates success

    def __soft_inlier_count(self,pts,planes):
        '''
        Soft inlier count for a given plane and a given set of points.

        hs -- Four parameters of the plane
        x -- vector of x values
        y -- vector of y values
        z -- vector of z values
        '''

        # point plane distances
        dists = torch.abs(planes[:,0].view(-1,1)*pts[:,0]+planes[:,1].view(-1,1)*pts[:,1]+planes[:,2].view(-1,1)*pts[:,2]+planes[:,3].view(-1,1))#.cuda()
        # print("distances")
        # print(dists[0])
        # print(dists)
        # print("fenmu")
        # print(torch.sqrt(planes[:,0].view(-1,1)**2+planes[:,1].view(-1,1)**2+planes[:,2].view(-1,1)**2)[0])
        # print(torch.sqrt(planes[:,0].view(-1,1)**2+planes[:,1].view(-1,1)**2+planes[:,2].view(-1,1)**2))
        dists = dists / torch.sqrt(planes[:,0].view(-1,1)**2+planes[:,1].view(-1,1)**2+planes[:,2].view(-1,1)**2)#.cuda()
        # print("除分母后")
        # print(dists[0])
        # print(dists)
        # # soft inliers,越远的点贡献的分数越低
        #weights = gaussian(dists)
        dists = 1- torch.sigmoid(self.inlier_beta * (dists- self.inlier_thresh) )
        #score =  1 - torch.sigmoid(self.inlier_beta * (loss - self.inlier_thresh))
        #score = score*50
        # print("sigmoid")
        # print(dists[0])
        # print(dists)
        #dists =dists*weights
        #print(dists)
        score = torch.sum(dists,1)#.cuda()
        # print("得分")
        # print(score)
        #返回的dists实际上是一组权重
        # for i in range(len(dists)):
        #     if dists[i][0] ==0:
        #         print(dists[i])
        #         print(planes[i])
        return score#, dists

    def __refine_hyp(self, pts, weights):
        '''
        Refinement by weighted PCA.

        Fits a plane minimizing errors in x, y and z    

        x -- vector of x values
        y -- vector of y values
        z -- vector of z values
        weights -- vector of weights (1 per point)        
        '''
        
        weights=weights.view(weights.size(0),-1,1)
        x=weights*pts
        x=x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.sum(2)
        x = x.view(-1,512)
        x = F.relu(self.bn5(self.fc0(x)))
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        # print(pts)
        # print(weights)
        # ws = weights.sum(1)
        # xm = ((pts[:,0] * weights).sum(1) / ws).view(-1,1,1)
        # ym = ((pts[:,1] * weights).sum(1) / ws).view(-1,1,1)
        # zm = ((pts[:,2] * weights).sum(1) / ws).view(-1,1,1)

        # A=[xm,ym,zm]
        # #B=[xm,ym,zm]
        # cov = torch.empty([weights.size(0),3,3]).cuda()
        
        # #cov=[]
        # for i in range(3):
        #     #cov1 = []
        #     for j in range(3):
        #         cov[:,i,j]=((pts[:,i]-A[i][:,0])*(pts[:,j]-A[j][:,0])*weights).sum(1)
        # hs = []
        
        # for c in range(len(cov)):
        #     #print(mcov)
        #     #if iszero(cov[c][0][0]):
        #         #print("weights",weights[c])
        #         #hs.append(torch.zeros(3).cuda())
            
        #     vec,_,_ = torch.svd(cov[c])
        #     hs.append(vec[:,2])
        # hs = torch.stack(hs)
        #print(hs)
        #         s = torch.mul(A[i],A[j])
        #         s = (s * weights).sum()
        #         #cov[i][j] = s
        #         cov1.append(s)
        #     cov1=torch.stack(cov1)#.cuda()
        #     cov.append(cov1)
        # cov =torch.stack(cov)#.cuda()  
        # #print(cov)
        # vec,_,_ = torch.svd(cov)
        # vec=vec
        # #hs = torch.zeros([4])
        # hs=[]
        # for i in range(3):
        #     hs.append( vec[i][2])
        #     #hs[3] = (hs[3]-vec[i][2] * B[i])
        # hs.append((-hs[0]*xm-hs[1]*ym-hs[2]*zm))
        # hs = torch.stack(hs)#.cuda()
        return hs
        

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

        # working on CPU because of many, small matrices
        #prediction = prediction
        #prediction=x.transpose(1,2)
        #print("before input",x,x.size())
        #print(x)
        #print(x.size())
        
        prediction,_,_,_=self.pcpnet(x)
        #print("after input",prediction,prediction.size())
        #print(prediction)
        #print(prediction.size())
        batch_size = prediction.size(0)
        #batch_size = x.size(0)
        invalid_size = 0
        avg_exp_loss = 0#torch.zeros(1).cuda() # expected loss
        avg_top_loss = 0#torch.zeros(1).cuda() # loss of best hypothesis

        #self.est_parameters = torch.zeros(batch_size,3 ).cuda() # estimated lines
        self.est_losses = torch.zeros(batch_size).cuda() # loss of estimated lines
        #self.batch_inliers = torch.zeros(batch_size, prediction.size(2)) # (soft) inliers for estimated lines
        #with torchsnooper.snoop():
        for b in range(0, batch_size):

            #hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
            #hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis
            #hyp_losses =[]
            #hyp_scores =[]
            #max_score = 0#torch.zeros(1).cuda()     # score of best hypothesis
            #pts = x[b]
            # ind = torch.nonzero(tpts[b])
            # if ind.size(0)%3==0:
            #     ind = ind[:,0].view(-1,3)[:,0]
            # else:
            #     offset = torch.zeros(3-ind.size(0)%3,2).long().cuda()
            #     ind=torch.cat((ind,offset),0)
            #     ind = ind[:,0].view(-1,3)[:,0]
            # if len(ind)< 6:
            #     invalid_size = invalid_size +1
            #     continue
            #print("进入的点",pts[ind])
            # pts = pts.view(1,500,3)
            # pts = pts.transpose(2,1)
            # pts = self.pcpnet(pts)
            #pts = pts.view(64,3)
            
            pts = prediction[b]
            #print(pts)
            index = torch.nonzero(pts)

            #index = index[:,0].view(-1,3)[:,0]

            if index.size(0)%3==0:
                index = index[:,0].view(-1,3)[:,0]
            else:
                offset = torch.zeros(3-index.size(0)%3,2).long().cuda()
                index=torch.cat((index,offset),0)
                index = index[:,0].view(-1,3)[:,0]
            pts=pts[index,:]
            #print("出网络的点",pts)
            if pts.size(0)<4:
                invalid_size = invalid_size +1
                continue
            #x,y,z=torch.split(pts,[1,1,1],1)
            #x=x.view(-1)
            #y=y.view(-1)
            #z=z.view(-1)
            #print(x,y,z)
            #for h in range(0, self.hyps):    

                # === step 1: sample hypothesis ===========================
            planes,valid = self.__sample_hyp(pts,self.hyps)
            if not valid: 
                invalid_size = invalid_size +1
                continue # skip invalid hyps
                #print(hs)
                # === step 2: score hypothesis using soft inlier count ====
            
            score= self.__soft_inlier_count(pts,planes)
                #print(score)
                #print(inliers)
                # === step 3: refine hypothesis ===========================
            norplanes=[]
            norplanes.append(planes[:,0]/torch.sqrt(planes[:,0]**2+planes[:,1]**2+planes[:,2]**2))
            norplanes.append(planes[:,1]/torch.sqrt(planes[:,0]**2+planes[:,1]**2+planes[:,2]**2))
            norplanes.append(planes[:,2]/torch.sqrt(planes[:,0]**2+planes[:,1]**2+planes[:,2]**2))
            norplanes=torch.stack(norplanes)
            norplanes=norplanes.transpose(1,0)

            loss = compute_loss(
                norplanes,
                #hs, 
                target[b],
                    #output_loss_weight =output_loss_weight,
                normal_loss = self.normal_loss
                    
            )
            #hs = self.__refine_hyp(pts, inliers)
                #hs = hs
            
            # print("归一化后")
            # print(norplanes)
            # print("ground truth")
            # print(target[b])
        
            
            # # #     # === step 4: calculate loss of hypothesis ================
             
            # print("loss")
            # print(loss)
            # print("score")
            # print(score)
                # store results
            #    hyp_losses.append(loss)
             #   hyp_scores.append(score)
           # os.system("pause")
                # keep track of best hypothesis so far
                #if score > max_score:
            maxindex = torch.argmax(score)
            self.est_losses[b] = loss[maxindex]
         #           self.est_parameters = hs[0:3]
                    #self.batch_inliers = inliers
            #hyp_losses =torch.stack(hyp_losses)#.cuda()
            #hyp_scores =torch.stack(hyp_scores)#.cuda()
            # === step 5: calculate the expectation ===========================

            #softmax distribution from hypotheses scores            
            score = F.softmax(self.inlier_alpha * score, 0)
            # print("softmax")
            # print(score)
            # expectation of loss
            exp_loss = torch.sum(loss * score)#.cuda()
            # print("exp_loss")
            # print(exp_loss)
            avg_exp_loss = (avg_exp_loss + exp_loss)#.cuda()
            #print(exp_loss)
            avg_top_loss = (avg_top_loss + self.est_losses[b])#.cuda()
            #print(avg_exp_loss)
    
        return avg_exp_loss / (batch_size-invalid_size), avg_top_loss / (batch_size-invalid_size)

