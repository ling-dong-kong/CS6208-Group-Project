import os
import glob
import h5py
import numpy as np

from torch.utils.data import Dataset
from utils import PointWOLF


def load_h5(h5_name):
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data_cls(data_root, partition):
    download_modelnet40()
    
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    # for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
    for h5_name in glob.glob(os.path.join(data_root, f'ply_data_{partition}*.h5')):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')  # (2048, 2048, 3)
        label = f['label'][:].astype('int64')  # (2048, 1)
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)    # train: (9840, 2048, 3); test: (2468, 2048, 3)
    all_label = np.concatenate(all_label, axis=0)  # train: (9840, 1)      ; test: (2468, 1)
    
    return all_data, all_label
 

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x, z)
    
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, data_root, num_points, partition='train', if_dual=False, ratio=0.):
        self.data, self.label = load_data_cls(data_root, partition)
        self.num_points = num_points
        self.partition = partition
        self.PointWOLF = PointWOLF(if_aug=True)
        self.if_dual = if_dual
        self.ratio = ratio

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
            
        if not self.if_dual:
            if self.partition == 'train':
                np.random.shuffle(pointcloud)
                _, pointcloud = self.PointWOLF(pointcloud)
                pointcloud = translate_pointcloud(pointcloud)
            
            return pointcloud, label
        
        else:
            pointcloud2 = pointcloud.copy()
            max_num_drop = int(len(pointcloud2) * self.ratio)
            self.points_to_drop = np.random.choice(np.arange(len(pointcloud2)), size=max_num_drop, replace=False)
            pointcloud2 = np.delete(pointcloud2, self.points_to_drop, axis=0)
            label2 = label.copy()
            
            if self.partition == 'train':
                np.random.shuffle(pointcloud)
                _, pointcloud = self.PointWOLF(pointcloud)
                pointcloud = translate_pointcloud(pointcloud)
                
                np.random.shuffle(pointcloud2)
                _, pointcloud2 = self.PointWOLF(pointcloud2)
                pointcloud2 = translate_pointcloud(pointcloud2)
            
            return pointcloud, label, pointcloud2, label2

    def __len__(self):
        return self.data.shape[0]


class ModelNetC(Dataset):
    def __init__(self, data_root, split):
        h5_path = os.path.join(data_root, split + '.h5')
        self.data, self.label = load_h5(h5_path)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)

