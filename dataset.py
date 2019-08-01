from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial

from pointnet2.utils.pointnet2_modules import PointnetSAModule
# do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
# so modifying the points would make the kdtree give incorrect results
def load_shape(point_filename, normals_filename,  pidx_filename):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
    else:
        normals = None


    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals, pidx=patch_indices)

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():
    def __init__(self, pts, kdtree, normals=None,  pidx=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter ==self.counter + 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, shape_list_filename, patch_radius, points_per_patch,
                 seed=None, identical_epochs=False, center='point',  cache_capacity=1,  sparse_patches=False):

        # initialize parameters
        self.root = root
        self.shape_list_filename = shape_list_filename

        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs

        self.sparse_patches = sparse_patches
        self.center = center

    
        self.seed = seed


        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)


            normals_filename = os.path.join(self.root, shape_name+'.normals')
            normals = np.loadtxt(normals_filename).astype('float32')
            np.save(normals_filename+'.npy', normals)


            if self.sparse_patches:
                pidx_filename = os.path.join(self.root, shape_name+'.pidx')
                patch_indices = np.loadtxt(pidx_filename).astype('int')
                np.save(pidx_filename+'.npy', patch_indices)
            else:
                pidx_filename = None

            shape = self.shape_cache.get(shape_ind)

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))#对角线
            self.patch_radius_absolute.append(bbdiag * self.patch_radius[0])#因为在这里我添加的不是list，就只是一个半径的值

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        # get neighboring points (within euclidean distance patch_radius)
        patch_pts = torch.zeros(self.points_per_patch, 3, dtype=torch.float)#？？？？我不知道为什么要乘个绝对半径,不用乘，就只有一个
        # patch_pts_valid = torch.ByteTensor(self.points_per_patch*len(self.patch_radius_absolute[shape_ind])).zero_()


        #for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
        patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], self.patch_radius[0]))
        #print("刚取出来的时候")
        #print(patch_point_inds)
            # optionally always pick the same points for a given patch index (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed((self.seed + index) % (2**32))

        point_count = min(self.points_per_patch, len(patch_point_inds))

            # if there are too many neighbors, pick a random subset
        if point_count < len(patch_point_inds):
            patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]
        #print("可能被随机减少了一部分后")
        #print(patch_point_inds)
        start = 0
        end = point_count

        if end >3:
            valid =1
        else:
            valid =0
        if  end < 64:
            modelrad = self.patch_radius_absolute[shape_ind]*0.2
        elif end < 128:
            modelrad = self.patch_radius_absolute[shape_ind]*0.25
        elif end < 256:
            modelrad = self.patch_radius_absolute[shape_ind]*0.3
        else:
            modelrad = self.patch_radius_absolute[shape_ind]*0.4 
            # convert points to torch tensors
        
        patch_pts[start:end, :] = torch.from_numpy(shape.pts[patch_point_inds, :])
        patch_pts =patch_pts.cuda()
       
        # print("转成tensor之后",patch_pts[0:50])
        model=PointnetSAModule(
                npoint=32,
                radius=modelrad,
                nsample=16,
                
                use_xyz=True,
            ).cuda()
        patch_pts = model(patch_pts.view(1,512,3))
        patch_pts = patch_pts.view(3,512).transpose(1,0)
        
        #print("pooling之后",patch_pts[:,0])
            # center patch (central point at origin - but avoid changing padded zeros)
        if self.center == 'mean':
            patch_pts = patch_pts - patch_pts.mean(0)
        elif self.center == 'point':
            patch_pts = patch_pts - torch.from_numpy(shape.pts[center_point_ind, :]).cuda()
        elif self.center == 'none':
            pass # no centering
        else:
            raise ValueError('Unknown patch centering option: %s' % (self.center))
      #  print("减去中心点后")
       # print(patch_pts)
            # normalize size of patch (scale with 1 / patch radius)
        patch_pts = patch_pts / self.patch_radius_absolute[shape_ind]
        # print("zhengze之后",patch_pts[:,0])
       # print("除半径后")
       # print(patch_pts)

        
        patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])
        #print("相匹配的normal")
        #print(patch_normal)

        return (patch_pts,) + (patch_normal,)+(valid,)


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') 

        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None
        return load_shape(point_filename, normals_filename,  pidx_filename)
