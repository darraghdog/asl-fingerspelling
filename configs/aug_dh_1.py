import random
from albumentations.core.transforms_interface import BasicTransform
from torch.nn import functional as F
from albumentations import Compose, random_utils
import torch
import numpy as np
import math

import random


def crop_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # Crop
        if mode == "start":
            data = data[:max_len]
        else:
            offset = np.abs(diff) // 2
            data = data[offset: offset + max_len]
        return data
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2])) * coef
    data = torch.cat([data, padding])
    return data

          
class Resample(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(0.8,1.2),
        always_apply=False,
        p=0.5,
    ):
        super(Resample, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, sample_rate=1., **params):
        length = data.shape[0]
        new_size = max(int(length * sample_rate),1)
        new_x = F.interpolate(data.permute(1,2,0),new_size).permute(2,0,1)
        return new_x

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
          
class Mirror(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        always_apply=False,
        p=0.5,
    ):
        super(Mirror, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        self.part_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in '_face_ _left_hand_ _right_hand_'.split()]

    def apply(self, data, **params):
        
        for frame in range(len(data)):
            m = data[frame].clone()
            '''
            # Flip pose
            mpo = m[self.cfg.input_type_idx[2]]
            pose_x_mean = mpo[:,0].mean()
            r_side, l_side = mpo[self.cfg.r_pose_idx, 0], mpo[self.cfg.l_pose_idx, 0]
            r_side = r_side + 2 * (pose_x_mean - r_side )
            l_side = l_side + 2 * (pose_x_mean - l_side )
            mpo[self.cfg.l_pose_idx, 0], mpo[self.cfg.r_pose_idx, 0] = r_side, l_side
            mpo[self.cfg.l_pose_idx, 1], mpo[self.cfg.r_pose_idx, 1] = \
                mpo[self.cfg.r_pose_idx, 1], mpo[self.cfg.l_pose_idx, 1]
            mpo[self.cfg.l_pose_idx, 2], mpo[self.cfg.r_pose_idx, 2] = \
                    mpo[self.cfg.r_pose_idx, 2], mpo[self.cfg.l_pose_idx, 2]
            m[self.cfg.input_type_idx[2]] = mpo
            '''
            face_x_mean =  m[self.part_indices[0]][:,0].mean()
            # Flip left and right hand
            mlh = m[self.part_indices[1]]
            if m[self.part_indices[1]].sum()!=0:
                mlh[:,0] = mlh[:,0].mean() + (mlh[:,0].mean() - mlh[:,0]) 
                mlh[:,0] += 2 * (face_x_mean - mlh[:,0].mean() )
            
            mrh = m[self.part_indices[2]]
            if m[self.part_indices[2]].sum()!=0:
                mrh[:,0] = mrh[:,0].mean() + (mrh[:,0].mean() - mrh[:,0]) 
                mrh[:,0] += 2 * (face_x_mean - mrh[:,0].mean() )
            
            m[self.part_indices[1]] = mrh
            m[self.part_indices[2]] = mlh
            
            # Assign the flipped coordinates back - assume we dont need to flip lips
            data[frame] = m
        
        return data

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()
    
    @property
    def targets(self):
        return {"image": self.apply}
    

class TemporalCrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        length=384,
        always_apply=False,
        p=0.5,
    ):
        super(TemporalCrop, self).__init__(always_apply, p)

        self.length = length

    def apply(self, data, length=384,offset_01=0.5, **params):
        l = data.shape[0]
        max_l = np.clip(l-length,1,length)
        offset = int(offset_01 * max_l)
        data = data[offset:offset+length]
        return data

    def get_params(self):
        return {"offset_01": random.uniform(0, 1)}

    def get_transform_init_args_names(self):
        return ("length", )
    
    @property
    def targets(self):
        return {"image": self.apply}    

    
class TemporalMask(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        mask_value=float('nan'),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalMask, self).__init__(always_apply, p)

        self.size = size
        self.mask_value = mask_value

    def apply(self, data, mask_size=0.3,mask_offset_01=0.2, mask_value=float('nan'), **params):
        l = data.shape[0]
        mask_size = int(l * mask_size)
        max_mask = np.clip(l-mask_size,1,l)
        mask_offset = int(mask_offset_01 * max_mask)
        x_new = data.contiguous()
        x_new[mask_offset:mask_offset+mask_size] = torch.tensor(mask_value)
        return x_new

    def get_params(self):
        return {"mask_size": random.uniform(self.size[0], self.size[1]),
                'mask_offset_01':random.uniform(0, 1),
                'mask_value':self.mask_value,}

    def get_transform_init_args_names(self):
        return ("size","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class FingerDrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        n_fingers, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FingerDrop, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        self.finger_indices = np.reshape(hand_indices[:,1:], (-1, 4))
        self.n_fingers = n_fingers
        self.mask_value = mask_value

    def apply(self, data, **params):
        x_new = data.contiguous()
        
        # Drop fingers
        fidx = np.random.randint(len(self.finger_indices), size=self.n_fingers)
        drop_indices = self.finger_indices[fidx].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  


class ArmsDrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        drop_proba, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(ArmsDrop, self).__init__(always_apply, p)
        arm_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_pose'.split()]
        arm_indices = np.array(arm_indices)
        self.arm_indices = np.reshape(arm_indices, (2, -1))
        self.mask_value = mask_value
        self.drop_proba = drop_proba

    def apply(self, data, **params):
        x_new = data.contiguous()
        
        # Drop fingers
        for drop_indices in self.arm_indices:
            if np.random.random() < self.drop_proba:
                x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ( "drop_proba","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  


def rotate_points(pos, center, alpha):
    radian = alpha / 180 * np.pi
    rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]])
    translated_points = (pos - center).reshape(-1, 2)
    rotated_points = np.dot(rotation_matrix, translated_points.T).T.reshape(*pos.shape)
    rotated_pos = rotated_points + center
    return rotated_pos

# https://github.com/ffs333/2nd_place_GISLR/blob/89b4e7383d4bc898ffbee6fc93757f3893e9bce8/GISLR_utils/transformer_code/utils/augmentations.py#L173

class FingerTreeRotate(BasicTransform):
    """
    rotates the roots of fingers - from 2nd place solution of first comp
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        degree: degree of rotation of finger around the center 
        joint_prob : probability of each finger root being rotated. 
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        degree=(-4, 4), 
        joint_prob=0.15,
        always_apply=False,
        mask_value=float('nan'),
        p=1.0,
    ):
        super(FingerTreeRotate, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        self.lhand_indices = [t for t,l in enumerate(landmarks) if 'x_left_' in l]
        self.rhand_indices = [t for t,l in enumerate(landmarks) if 'x_right_' in l]
        self.degree = degree
        self.joint_prob = joint_prob
        HAND_ROUTES = [
            [0, *range(1, 5)], 
            [0, *range(5, 9)], 
            [0, *range(9, 13)], 
            [0, *range(13, 17)], 
            [0, *range(17, 21)],
        ]
        self.HAND_TREES = sum([[np.array(route[i:]) for i in range(len(route) - 1)] for route in HAND_ROUTES], [])
        self.mask_value = mask_value
        
    def apply(self, data, **params):
        
        x_new = data.contiguous()#.numpy()
        
        x_lh, x_rh = data[:,self.lhand_indices].contiguous(), data[:,self.rhand_indices].contiguous()
        x_lh, x_rh = x_lh.numpy(), x_rh.numpy()
        mask_lh, mask_rh = x_lh.sum((1,2))==self.mask_value, x_rh.sum((1,2))==self.mask_value
        
        for tree in self.HAND_TREES:
            if x_rh.sum()!=self.mask_value: 
                if np.random.rand() < self.joint_prob:
                    alpha = np.random.uniform(*self.degree)
                    
                    center = x_rh[:,tree[0:1],:2]
                    x_rh[:,tree[1:],:2] = rotate_points(x_rh[:,tree[1:],:2], center, alpha)
                x_rh[mask_rh] = self.mask_value

            if x_lh.sum()!=self.mask_value: 
                if np.random.rand() < self.joint_prob:
                    alpha = np.random.uniform(*self.degree)
                    center = x_lh[:,tree[0:1],:2]
                    x_lh[:,tree[1:],:2] = rotate_points(x_lh[:,tree[1:],:2], center, alpha)
                x_lh[mask_lh] = self.mask_value
                
        #print(x_new[:,self.rhand_indices].sum(), x_rh.sum())
        #print(x_new[:,self.lhand_indices].sum(), x_lh.sum())
        
        x_new[:,self.rhand_indices] = torch.from_numpy(x_rh)
        x_new[:,self.lhand_indices] = torch.from_numpy(x_lh)
        
        return x_new

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ( "degree", "joint_prob", "mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  


class FingerPartsDrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        n_fingers, 
        drop_n_nodes,
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FingerPartsDrop, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        self.finger_indices = np.reshape(hand_indices[:,1:], (-1, 4))
        self.n_fingers = n_fingers
        self.drop_n_nodes =  drop_n_nodes
        self.mask_value = mask_value

    def apply(self, data, **params):
        x_new = data.contiguous()
        
        # Drop fingers
        fidx = np.random.randint(len(self.finger_indices), size=self.n_fingers)
        drop_indices = self.finger_indices[fidx][:,-self.drop_n_nodes:].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  

import random
import typing

class OneOf(BasicTransform):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, 
                always_apply=False,
                p=1.0,):
        super(OneOf, self).__init__(always_apply, p)
        self.transforms = transforms
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:

        if self.transforms_ps and random.random() < self.p:
            idx: int = np.random.choice(range(len(self.transforms)), p=self.transforms_ps, size = 1)[0]
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data
    
    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()
    
    @property
    def targets(self):
        return {"image": self.apply}  

    
    
# landmarks = self.xyz_landmarks
class BodyPartCutout(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        ratio : likelihood of applying it to each body part
        size : max region size on 1d axis to apply it
        num_holes : num regions to apply it to

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        ratio = 0.4, 
        size = 0.15,
        num_holes=5,
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(BodyPartCutout, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        self.part_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in '_face_ _left_hand_ _right_hand_'.split()]
        self.ratio = ratio
        self.size = size
        self.num_holes = num_holes
        self.mask_value = mask_value

    def apply(self, data, **params):
        x_new = data.contiguous()
        for iddx in self.part_indices:
            m_len = len(x_new.view(-1, x_new.shape[-1]))
            m_nonnan_len = len(x_new[~torch.isnan(x_new)].view(-1, x_new.shape[-1])[:,0])
            if m_nonnan_len / m_len < 0.3:
                continue
            if np.random.random() < self.ratio:
                m = data[:, iddx]
                width = len(m)
                for _n in range(self.num_holes):
                    x = random.randint(0, width)
                    x1 = int(np.clip(x - (self.size * width) // 2, 0, width))
                    x2 = int(np.clip(x1 +  (self.size * width), 0, width))
                    x_new[x1:x2, iddx] = torch.tensor(self.mask_value)
        return x_new

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ( "size","ratio","num_holes","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
# def spatial_mask(x, size=(0.5,1.), mask_value=float('nan')):
#     mask_offset_y = np.random.uniform(x[...,1].min().item(),x[...,1].max().item())
#     mask_offset_x = np.random.uniform(x[...,0].min().item(),x[...,0].max().item())
#     mask_size = np.random.uniform(size[0],size[1])
#     mask_x = (mask_offset_x<x[...,0]) & (x[...,0] < mask_offset_x + mask_size)
#     mask_y = (mask_offset_y<x[...,1]) & (x[...,1] < mask_offset_y + mask_size)
#     mask = mask_x & mask_y
#     x_new = x.contiguous()
#     x_new = x * (1-mask[:,:,None].float()) #+ mask_value[:,:,None] * mask_value
#     return x_new
    
    
class SpatialMask(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.5,1.), 
        mask_value=float('nan'),
        mode = 'abolute',
        always_apply=False,
        p=0.5,
    ):
        super(SpatialMask, self).__init__(always_apply, p)

        self.size = size
        self.mask_value = mask_value
        self.mode = mode

    def apply(self, data, mask_size=0.75, offset_x_01=0.2, offset_y_01=0.2,mask_value=float('nan'), **params):
        # mask_size absolute width 
        
        
        
        #fill na makes it easier with min and max
        data0 = data.contiguous()
        data0[torch.isnan(data0)] = 0
        
        x_min, x_max = data0[...,0].min().item(), data0[...,0].max().item() 
        y_min, y_max = data0[...,1].min().item(), data0[...,1].max().item() 
        
        if self.mode == 'relative':
            mask_size_x = mask_size * (x_max - x_min)
            mask_size_y = mask_size * (y_max - y_min)
        else:
            mask_size_x = mask_size 
            mask_size_y = mask_size             

        mask_offset_x = offset_x_01 * (x_max - x_min) + x_min
        mask_offset_y = offset_y_01 * (y_max - y_min) + y_min
        
        mask_x = (mask_offset_x<data0[...,0]) & (data0[...,0] < mask_offset_x + mask_size_x)
        mask_y = (mask_offset_y<data0[...,1]) & (data0[...,1] < mask_offset_y + mask_size_y)
        
        mask = mask_x & mask_y
        x_new = data.contiguous() * (1-mask[:,:,None].float()) + mask[:,:,None] * mask_value
        return data

    def get_params(self):
        params = {"offset_x_01": random.uniform(0, 1)}
        params['offset_y_01'] = random.uniform(0, 1)
        params['mask_size'] = random.uniform(self.size[0], self.size[1])
        params['mask_value'] = self.mask_value
        return params

    def get_transform_init_args_names(self):
        return ("size", "mask_value","mode")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class SpatialNoise(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        noise_range=(-0.05,0.05), 
        always_apply=False,
        p=0.5,
    ):
        super(SpatialNoise, self).__init__(always_apply, p)

        self.noise_range = noise_range

    def apply(self, data, noise, **params):
        # mask_size absolute width 
        
        data = data + torch.tensor(noise, dtype=data.dtype)
        return data
    
    def get_params_dependent_on_targets(self, params):
        data = params["image"]
        noise = random_utils.uniform(self.noise_range[0],self.noise_range[1],data.shape)

        return {"noise": noise}

    def get_transform_init_args_names(self):
        return ("noise_range",)
    
    @property
    def targets_as_params(self):
        return ["image"]
    
    @property
    def targets(self):
        return {"image": self.apply}
    
def spatial_random_affine(data,scale=None,shear=None,shift=None,degree=None,center=(0,0)):
    
    data_tmp = None
    
    #if input is xyz, split off z and re-attach later
    if data.shape[-1] == 3:
        data_tmp = data[...,2:]
        data = data[...,:2]
        
    center = torch.tensor(center)
    
    if scale is not None:
        data = data * scale
        
    if shear is not None:
        shear_x, shear_y = shear
        shear_mat = torch.tensor([[1.,shear_x],
                                  [shear_y,1.]])    
        data = data @ shear_mat
        center = center + torch.tensor([shear_y, shear_x])
        
    if degree is not None:
        data -= center
        radian = degree/180*np.pi
        c = math.cos(radian)
        s = math.sin(radian)
        
        rotate_mat = torch.tensor([[c,s],
                                   [-s, c]])
        
        data = data @ rotate_mat
        data = data + center
        
    if shift is not None:
        data = data + shift
                          
    if data_tmp is not None:
        data = torch.cat([data,data_tmp],axis=-1)
        
    return data    
    
class SpatialAffine(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        scale (float, float) or None
        
        
        
    
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        scale  = None,
        shear = None,
        shift  = None,
        degree = None,
        center_xy = (0,0),
        always_apply=False,
        p=0.5,
    ):
        super(SpatialAffine, self).__init__(always_apply, p)

        self.scale  = scale
        self.shear  = shear
        self.shift  = shift
        self.degree  = degree
        self.center_xy = center_xy

    def apply(self, data, scale=None,shear=None,shift=None,degree=None,center=(0,0), **params):
        
        new_x = spatial_random_affine(data,scale=scale,shear=shear,shift=shift,degree=degree,center=center)
        return new_x

    def get_params(self):
        params = {'scale':None, 'shear':None, 'shift':None, 'degree':None,'center_xy':self.center_xy}
        if self.scale:
            params['scale']= random.uniform(self.scale[0], self.scale[1])
        if self.shear:
            
            shear_x = shear_y = random.uniform(self.shear[0],self.shear[1])
            if random.uniform(0,1) < 0.5:
                shear_x = 0.
            else:
                shear_y = 0.     
            params['shear']= (shear_x, shear_y)
        if self.shift:
            params['shift']= random.uniform(self.shift[0], self.shift[1])
        if self.degree:
            params['degree']= random.uniform(self.degree[0], self.degree[1])
        
        return params

    def get_transform_init_args_names(self):
        return ("scale", "shear", "shift", "degree")
    
    @property
    def targets(self):
        return {"image": self.apply}

class Resample2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks,
        sample_rate=(0.8,1.2),
        always_apply=False,
        p=0.5,
    ):
        super(Resample2, self).__init__(always_apply, p)
        
        lmks = [i for i in landmarks if 'x_' in i]
        self.lmk_parts = [[t for t,i in enumerate(lmks) if j in i] \
                          for j in 'x_left_hand x_right_hand x_face x_pose'.split() ]
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, sample_rate=1., **params):
        
        for lmk_idx in self.lmk_parts:
            to_nan_idx = (torch.where(data[:,lmk_idx].sum((1,2)) == 0.)[0])
            data[to_nan_idx.unsqueeze(1), torch.tensor(lmk_idx)] = torch.nan
        
        length = data.shape[0]
        new_size = max(int(length * sample_rate),1)
        new_x = F.interpolate(data.permute(1,2,0),new_size).permute(2,0,1)
        
        new_x = torch.nan_to_num(new_x, nan=0.0) 
        
        return new_x

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
class SpatialAffine2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        scale (float, float) or None
        
        
        
    
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks,
        scale  = None,
        shear = None,
        shift  = None,
        degree = None,
        center_xy = (0,0),
        always_apply=False,
        p=0.5,
    ):
        super(SpatialAffine2, self).__init__(always_apply, p)
        
        lmks = [i for i in landmarks if 'x_' in i]
        
        self.lmk_parts = [[t for t,i in enumerate(lmks) if j in i] \
                          for j in 'x_left_hand x_right_hand x_face x_pose'.split() ]
        self.scale  = scale
        self.shear  = shear
        self.shift  = shift
        self.degree  = degree
        self.center_xy = center_xy

    def apply(self, data, scale=None,shear=None,shift=None,degree=None,center=(0,0), **params):
        
        for lmk_idx in self.lmk_parts:
            to_nan_idx = (torch.where(data[:,lmk_idx].sum((1,2)) == 0.)[0])
            data[to_nan_idx.unsqueeze(1), torch.tensor(lmk_idx)] = torch.nan
            break
        new_x = spatial_random_affine(data,scale=scale,shear=shear,shift=shift,degree=degree,center=center)
        return new_x

    def get_params(self):
        params = {'scale':None, 'shear':None, 'shift':None, 'degree':None,'center_xy':self.center_xy}
        if self.scale:
            params['scale']= random.uniform(self.scale[0], self.scale[1])
        if self.shear:
            
            shear_x = shear_y = random.uniform(self.shear[0],self.shear[1])
            if random.uniform(0,1) < 0.5:
                shear_x = 0.
            else:
                shear_y = 0.     
            params['shear']= (shear_x, shear_y)
        if self.shift:
            params['shift']= random.uniform(self.shift[0], self.shift[1])
        if self.degree:
            params['degree']= random.uniform(self.degree[0], self.degree[1])
        
        return params

    def get_transform_init_args_names(self):
        return ("scale", "shear", "shift", "degree")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
'''
n_fingers = (2,6); landmarks = xyz_landmarks; mask_value=0.
self = BasicTransform()
'''
class FingersDrop2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        n_fingers, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FingersDrop2, self).__init__(always_apply, p)
        
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        
        self.finger_indices_type1 = np.reshape(hand_indices[:,1:], (-1, 4))
        
        self.finger_indices_type2 = self.finger_indices_type1[:,1:]
        
        finger_root = np.array([40, 40, 40, 40, 40, 61, 61, 61, 61, 61])[:,None]
        self.finger_indices_type3 = np.concatenate((finger_root, self.finger_indices_type1),1)
        
        if type(n_fingers) == int:
            self.n_fingers = (n_fingers,n_fingers+1)
        else:
            self.n_fingers = n_fingers
        self.mask_value = mask_value

    def apply(self, data, fidx=None, **params):
        x_new = data.contiguous()
        
        # Drop fingers
#         n_fingers = np.random.randint(self.n_fingers[0])
#         fidx = np.random.randint(len(self.finger_indices), size=self.n_fingers)
        finger_indices = random.choice([self.finger_indices_type1,
                                            self.finger_indices_type2,
                                            self.finger_indices_type3])
        drop_indices = finger_indices[fidx].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        n_fingers = np.random.randint(self.n_fingers[0],self.n_fingers[1])
        fidx = np.random.randint(len(self.finger_indices_type1), size=self.n_fingers)
        params = {
                  'fidx':fidx}
        return params

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class PoseDrop2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(PoseDrop2, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        pose_indices = np.array([t for t,l in enumerate(landmarks) if 'pose' in l])
        pose_indices = pose_indices.reshape(-1, 2).T
        self.pose_indices_type1 = pose_indices[:,0:].T.reshape(-1)
        self.pose_indices_type2 = pose_indices[:,1:].T.reshape(-1)
        self.pose_indices_type3 = pose_indices[:,2:].T.reshape(-1)
        self.pose_indices_type4 = pose_indices[:,3:].T.reshape(-1)
                
        self.mask_value = mask_value

    def apply(self, data, **params):
        x_new = data.contiguous()
        
        pose_indices = random.choice([self.pose_indices_type1,
                                      self.pose_indices_type2,
                                      self.pose_indices_type3,
                                      self.pose_indices_type4])
        drop_indices = pose_indices.flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        #pidx = range(len(self.pose_indices)) #all pose idxs
        #params = {'pidx':pidx}
        params = {}
        return params

    def get_transform_init_args_names(self):
        return ( "mask_value",)
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class HandDrop2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(HandDrop2, self).__init__(always_apply, p)
        
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        self.hand_indices = np.array(hand_indices).flatten()
        self.mask_value = mask_value

    def apply(self, data, fidx=None, **params):
        x_new = data.contiguous()
        
        drop_indices = self.hand_indices
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        params = {}
        return params

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply} 
    
class ShiftParts(BasicTransform):
    """
    shifts parts over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        sample_shift=(1, 6),
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(ShiftParts, self).__init__(always_apply, p)
        
        rate_lower = sample_shift[0]
        rate_upper = sample_shift[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))
        
        self.rate_lower = rate_lower
        self.rate_upper = rate_upper
        
        landmarks = [i for i in landmarks if 'x_' in i]

        self.hand_indices = [t for t,l in enumerate(landmarks) if not any (j in l for j in ['x_face', 'x_pose'] )]
        self.face_indices = [t for t,l in enumerate(landmarks) if 'x_face' in l]
        self.pose_indices = [t for t,l in enumerate(landmarks) if 'x_pose' in l]
        self.mask_value = mask_value

    def apply(self, data, sample_rates_face, sample_rates_pose, fidx=None, **params):
        
        '''
        sample_rates_face, sample_rates_pose = 0.02, 0.05
        '''
        x_new = torch.zeros_like(data)#.contiguous()
        x_new[:] = self.mask_value
        
        shift_frames_face = int(round(sample_rates_face))
        shift_frames_pose = int(round(sample_rates_pose))
        
        if random.randint(1, 2)==1:
            x_new[shift_frames_face:,self.face_indices] = data[:-shift_frames_face,self.face_indices]
        else:
            x_new[:-shift_frames_face,self.face_indices] = data[shift_frames_face:,self.face_indices]
            
        
        if random.randint(1, 2)==1:
            x_new[shift_frames_pose:,self.pose_indices] = data[:-shift_frames_pose,self.pose_indices]
        else:
            x_new[:-shift_frames_pose,self.pose_indices] = data[shift_frames_pose:,self.pose_indices]
        
        x_new[:,self.hand_indices] = data[:,self.hand_indices]
        
        return x_new

    def get_params(self):
        sample_rates_pose = random.uniform(self.rate_lower, self.rate_upper)
        sample_rates_face = random.uniform(self.rate_lower, self.rate_upper)
        return {"sample_rates_face": sample_rates_face, "sample_rates_pose": sample_rates_pose,}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}


    
class ShiftFingers(BasicTransform):
    """
    shifts parts over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        sample_shift=(1, 6),
        n_fingers = (4, 10), 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(ShiftFingers, self).__init__(always_apply, p)
        
        rate_lower = sample_shift[0]
        rate_upper = sample_shift[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))
        
        self.rate_lower = rate_lower
        self.rate_upper = rate_upper
        
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        self.finger_indices = np.reshape(hand_indices[:,1:], (-1, 4))
        if type(n_fingers) == int:
            self.n_fingers = (n_fingers,n_fingers+1)
        else:
            self.n_fingers = n_fingers
        self.mask_value = mask_value

    def apply(self, data, sample_rates, fidx=None, **params):
        
        '''
        sample_rates_face, sample_rates_pose = 0.02, 0.05
        '''
        x_new = (data).contiguous()
        shift_frames = int(round(sample_rates))
        if random.randint(1, 2)==1:
            for iddx in self.finger_indices[fidx]:
                x_new[:,iddx] = self.mask_value
                
                x_new[shift_frames:,iddx] = data[:-shift_frames,iddx]
        else:
            for iddx in self.finger_indices[fidx]:
                x_new[:,iddx] = self.mask_value
                x_new[:-shift_frames,iddx] = data[shift_frames:,iddx]
        
        return x_new

    def get_params(self):
        
        sample_rates = random.uniform(self.rate_lower, self.rate_upper)
        
        n_fingers = np.random.randint(self.n_fingers[0],self.n_fingers[1])
        fidx = np.random.randint(len(self.finger_indices), size=n_fingers)
        params = {
                  'fidx':fidx}
        
        return {"sample_rates": sample_rates, 'fidx':fidx}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}

class StrobeHands(BasicTransform):
    """
    Blanks the signal and only keeps every n; should be the final augmentation
    
    Args:
        keep_every_n: integer

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        keep_every_n = (2,6),
        mask_value=float('nan'),
        always_apply=False,
        p=0.5,
    ):
        super(StrobeHands, self).__init__(always_apply, p)
        
        landmarks = [i for i in landmarks if 'x_' in i]
        non_hand_indices = [t for t,l in enumerate(landmarks) if not any(i in l for i in 'x_left_ x_right_'.split())]
        self.non_hand_indices = np.array(non_hand_indices)
        
        rate_lower = keep_every_n[0]
        rate_upper = keep_every_n[1]
        if not ((type(rate_lower)==int) & (type(rate_lower)==int)):
            raise ValueError("Input params should be type integer. Got: {}".format((rate_lower, rate_upper)))
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper
        self.mask_value = mask_value

    def apply(self, data, sample_rate=3, **params):
        
        new_x = torch.zeros_like(data)
        new_x[:] = self.mask_value
        start_posn = random.randint(0, sample_rate-1)
        new_x[start_posn::sample_rate] = data[start_posn::sample_rate]
        new_x[:,self.non_hand_indices] = data[:,self.non_hand_indices]
        
        return new_x

    def get_params(self):
        return {"sample_rate": int(round(random.uniform(self.rate_lower, self.rate_upper)))  }

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper", "mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
class StrobeHands2(BasicTransform):
    """
    Blanks the signal and only keeps every n; should be the final augmentation
    
    Args:
        keep_every_n: integer

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        keep_every_n = (2,6),
        mask_value=float('nan'),
        always_apply=False,
        p=0.5,
    ):
        super(StrobeHands2, self).__init__(always_apply, p)
        
        landmarks = [i for i in landmarks if 'x_' in i]
        non_hand_indices = [t for t,l in enumerate(landmarks) if not any(i in l for i in 'x_left_ x_right_'.split())]
        self.non_hand_indices = np.array(non_hand_indices)
        
        rate_lower = keep_every_n[0]
        rate_upper = keep_every_n[1]
        if not ((type(rate_lower)==int) & (type(rate_lower)==int)):
            raise ValueError("Input params should be type integer. Got: {}".format((rate_lower, rate_upper)))
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper
        self.mask_value = mask_value

    def apply(self, data, sample_rate=3, **params):
        
        new_x = torch.zeros_like(torch.cat([data] * 2))
        new_x[:] = self.mask_value
        #start_posn = random.randint(0, sample_rate-1)
        new_x[::sample_rate*2] = data[::sample_rate]
        new_x[0::2,self.non_hand_indices] = data[:,self.non_hand_indices]
        new_x[1::2,self.non_hand_indices] = data[:,self.non_hand_indices]
        
        return new_x

    def get_params(self):
        return {"sample_rate": int(round(random.uniform(self.rate_lower, self.rate_upper)))  }

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper", "mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
class FingerSetDrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        finger_set_size = 2,
        n_finger_sets = (1,3), 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FingerSetDrop, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        self.finger_indices = np.reshape(hand_indices[:,1:], (-1, 4))
        if type(n_finger_sets) == int:
            self.n_finger_sets = (n_finger_sets,n_finger_sets+1)
        else:
            self.n_finger_sets = n_finger_sets
        self.mask_value = mask_value
        self.finger_set_size = finger_set_size
        if self.finger_set_size == 2:
            self.finger_sets = np.stack((np.arange(4), np.arange(1,5))).T
            self.finger_sets = np.concatenate((self.finger_sets, self.finger_sets + 5))
        elif self.finger_set_size == 3:
            self.finger_sets = np.stack((np.arange(3), np.arange(1,4), np.arange(2,5))).T
            self.finger_sets = np.concatenate((self.finger_sets, self.finger_sets + 5))
        else:
            raise ValueError("Only finger sets of 2 & 3 supported. Got finger_set_size: {}".format(finger_set_size))
            
    def apply(self, data,fidx=None, **params):
        x_new = data.contiguous()
        
        # Drop fingers
#         n_fingers = np.random.randint(self.n_fingers[0])
#         fidx = np.random.randint(len(self.finger_indices), size=self.n_fingers)
        
        drop_set_indices = np.unique(self.finger_sets[fidx].flatten())
        drop_indices  = self.finger_indices[drop_set_indices].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        n_finger_sets = np.random.randint(self.n_finger_sets[0],1 + self.n_finger_sets[1])
        fidx = np.random.randint(len(self.finger_sets), size=n_finger_sets)
        params = {'fidx':fidx}
        return params

    def get_transform_init_args_names(self):
        return ( "finger_set_size", "n_finger_sets", "mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class HandCutOut(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        n_holes = (1,5), 
        hole_size = (1,4), 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(HandCutOut, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        self.hand_indices = np.reshape(hand_indices[:,1:], (-1, 4))
        
        if type(n_holes) == int:
            self.n_holes = (n_holes,n_holes+1)
        else:
            self.n_holes = n_holes
        if type(hole_size) == int:
            self.hole_size = (hole_size,hole_size+1)
        else:
            self.hole_size = hole_size
        self.mask_value = mask_value

    def apply(self, data, drop_indices = None,**params):
        x_new = data.contiguous()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        
        n_holes = np.random.randint(self.n_holes[0],self.n_holes[1])
        widths = [np.random.randint(self.hole_size[0],self.hole_size[1]) for _ in range(n_holes)]
        heights = [np.random.randint(self.hole_size[0],self.hole_size[1]) for _ in range(n_holes)]
        start_x = [np.random.choice(np.arange(self.hand_indices.shape[0]))  for _ in range(n_holes)]
        start_y = [np.random.choice(np.arange(self.hand_indices.shape[1]))  for _ in range(n_holes)]

        hidx = self.hand_indices.copy()
        if np.random.random()<0.5: hidx = np.fliplr(hidx)
        if np.random.random()<0.5: hidx = np.flipud(hidx)
        dropidx = [hidx[x_pos:x_pos+w, y_pos:y_pos+h] for x_pos,y_pos,w,h in zip(start_x, start_y, widths, heights)]
        dropidx = np.concatenate([i.flatten() for i in dropidx])
        dropidx = np.unique(dropidx)
        
        params = {'drop_indices': dropidx}
        return params

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  

class FingersBlur(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        n_fingers, 
        blur_weight,
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FingersBlur, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        self.finger_indices = np.reshape(hand_indices[:,1:], (2, -1, 4))
        if type(n_fingers) == int:
            self.n_fingers = (n_fingers,n_fingers+1)
        else:
            self.n_fingers = n_fingers
        if type(n_fingers) == float:
            self.blur_weight = (0,blur_weight)
        else:
            self.blur_weight = blur_weight
        self.mask_value = mask_value

    def apply(self, data,fidx=None, blur_weight = None, **params):
        x_new = data.contiguous()
        
        fidx_lh = fidx[fidx%2==0]//2
        fidx_rh = fidx[fidx%2==1]//2
        
        if len(fidx_lh)>0:
            x_new[:,self.finger_indices[0][fidx_lh+1].flatten()] = \
                x_new[:,self.finger_indices[0][fidx_lh].flatten()] * (blur_weight / 2) +  \
                + x_new[:,self.finger_indices[0][fidx_lh+1].flatten()] * (1 - blur_weight) +  \
                + x_new[:,self.finger_indices[0][fidx_lh+2].flatten()] * (blur_weight / 2)
            
        if len(fidx_rh)>0:
            x_new[:,self.finger_indices[1][fidx_rh+1].flatten()] = \
                x_new[:,self.finger_indices[1][fidx_rh].flatten()] * (blur_weight / 2) +  \
                + x_new[:,self.finger_indices[1][fidx_rh+1].flatten()] * (1 - blur_weight) +  \
                + x_new[:,self.finger_indices[1][fidx_rh+2].flatten()] * (blur_weight / 2)
        
        return x_new

    def get_params(self):
        n_fingers = np.random.randint(self.n_fingers[0],self.n_fingers[1])
        blur_weight = np.random.uniform(self.blur_weight[0],self.blur_weight[1])
        fidx = np.unique(np.random.randint((self.finger_indices.shape[1] - 2) * 2, size=self.n_fingers))
        params = {'fidx':fidx, 'blur_weight': blur_weight}
        
        return params

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    