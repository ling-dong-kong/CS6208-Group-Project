import numpy as np


class PointWOLF(object):
    
    def __init__(self, if_aug=True):
        if if_aug:
            self.num_anchor = 4
            self.sample_type = 'fps'
            self.sigma = 0.5
            
            self.w_R_range = 10.0
            self.w_S_range = 3.0
            self.w_T_range = 0.25

            self.R_range = (-abs(self.w_R_range), abs(self.w_R_range))
            self.S_range = (1.0, self.w_S_range)
            self.T_range = (-abs(self.w_T_range), abs(self.w_T_range))
        
    def __call__(self, pos):
        """
        input :
            pos([N,3])
            
        output : 
            pos([N,3]) : original pointcloud
            pos_new([N,3]) : Pointcloud augmneted by PointWOLF
        """
        M = self.num_anchor
        N, _ = pos.shape
        
        if self.sample_type == 'random':
            idx = np.random.choice(N, M)
        elif self.sample_type == 'fps':
            idx = self.fps(pos, M)
        
        pos_anchor = pos[idx] #(M,3), anchor point
        
        pos_repeat = np.expand_dims(pos,0).repeat(M, axis=0)  # (M, N, 3)
        pos_normalize = np.zeros_like(pos_repeat, dtype=pos.dtype)  # (M, N, 3)
        pos_normalize = pos_repeat - pos_anchor.reshape(M,-1,3)

        pos_transformed = self.local_transformaton(pos_normalize) # (M, N, 3)
        pos_transformed = pos_transformed + pos_anchor.reshape(M,-1,3)  # (M, N, 3)
        
        pos_new = self.kernel_regression(pos, pos_anchor, pos_transformed)
        pos_new = self.normalize(pos_new)
        
        return pos.astype('float32'), pos_new.astype('float32')
        
    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([N,3])
            pos_anchor([M,3])
            pos_transformed([M,N,3])
            
        output : 
            pos_new([N,3]) : Pointcloud after weighted local transformation 
        """
        M, N, _ = pos_transformed.shape
        
        sub = np.expand_dims(pos_anchor,1).repeat(N, axis=1) - np.expand_dims(pos,0).repeat(M, axis=0)  # (M, N, 3), d
        project_axis = self.get_random_axis(1)
        projection = np.expand_dims(project_axis, axis=1)*np.eye(3)  #(1, 3, 3)
        
        sub = sub @ projection  # (M, N, 3)
        sub = np.sqrt(((sub) ** 2).sum(2))  #(M, N)  
        
        weight = np.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))  # (M, N) 
        pos_new = (np.expand_dims(weight,2).repeat(3, axis=-1) * pos_transformed).sum(0) # (N, 3)
        pos_new = (pos_new / weight.sum(0, keepdims=True).T)  # normalize by weight
        
        return pos_new

    def fps(self, pos, npoint):
        """
        input : 
            pos([N,3])
            npoint(int)
            
        output : 
            centroids([npoints]) : index list for fps
        """
        N, _ = pos.shape
        centroids = np.zeros(npoint, dtype=np.int_) #(M)
        distance = np.ones(N, dtype=np.float64) * 1e10 #(N)
        farthest = np.random.randint(0, N, (1,), dtype=np.int_)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = ((pos - centroid)**2).sum(-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.argmax()
        return centroids
    
    def local_transformaton(self, pos_normalize):
        """
        input :
            pos([N,3]) 
            pos_normalize([M,N,3])
            
        output :
            pos_normalize([M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        M,N,_ = pos_normalize.shape
        transformation_dropout = np.random.binomial(1, 0.5, (M,3)) #(M,3)
        transformation_axis =self.get_random_axis(M) #(M,3)

        degree = np.pi * np.random.uniform(*self.R_range, size=(M,3)) / 180.0 * transformation_dropout[:,0:1] #(M,3), sampling from (-R_range, R_range) 
        
        scale = np.random.uniform(*self.S_range, size=(M,3)) * transformation_dropout[:,1:2] #(M,3), sampling from (1, S_range)
        scale = scale*transformation_axis
        scale = scale + 1*(scale==0) #Scaling factor must be larger than 1
        
        trl = np.random.uniform(*self.T_range, size=(M,3)) * transformation_dropout[:,2:3] #(M,3), sampling from (1, T_range)
        trl = trl*transformation_axis
        
        #Scaling Matrix
        S = np.expand_dims(scale, axis=1)*np.eye(3) # scailing factor to diagonal matrix (M,3) -> (M,3,3)
        #Rotation Matrix
        sin = np.sin(degree)
        cos = np.cos(degree)
        sx, sy, sz = sin[:,0], sin[:,1], sin[:,2]
        cx, cy, cz = cos[:,0], cos[:,1], cos[:,2]
        R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
             sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
             -sy, cy*sx, cy*cx], axis=1).reshape(M,3,3)
        
        pos_normalize = pos_normalize@R@S + trl.reshape(M,1,3)
        return pos_normalize
    
    def get_random_axis(self, n_axis):
        """
        input :
            n_axis(int)
            
        output :
            axis([n_axis,3]) : projection axis   
        """
        axis = np.random.randint(1,8, (n_axis)) # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz    
        m = 3 
        axis = (((axis[:,None] & (1 << np.arange(m)))) > 0).astype(int)
        return axis
    
    def normalize(self, pos):
        """
        input :
            pos([N,3])
        
        output :
            pos([N,3]) : normalized Pointcloud
        """
        pos = pos - pos.mean(axis=-2, keepdims=True)
        scale = (1 / np.sqrt((pos ** 2).sum(1)).max()) * 0.999999
        pos = scale * pos
        return pos
