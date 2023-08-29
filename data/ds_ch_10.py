import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch.nn as nn
import torch.nn.functional as F
import math

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

tr_collate_fn = None
val_collate_fn = None

class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def normalize(self,x):
        nonan = x[~torch.isnan(x)].view(-1, x.shape[-1])
        x = x - nonan.mean(0)[None, None, :]
        x = x / nonan.std(0, unbiased=False)[None, None, :]
        return x
    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x
        
    def forward(self, x):
        
        #seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3
        x = x.reshape(x.shape[0],3,-1).permute(0,2,1)
        
        # Normalize & fill nans
        x = self.normalize(x)
        x = self.fill_nans(x)
        return x

# augs

def flip(data, flip_array):
    
    data[:,:,0] = -data[:,:,0]
    data = data[:,flip_array]
    return data

def resample(x, rate=(0.8,1.2)):
    rate = np.random.uniform(rate[0], rate[1])
    length = x.shape[0]
    new_size = max(int(length * rate),1)
    new_x = F.interpolate(x.permute(1,2,0),new_size).permute(2,0,1)
    return new_x



def spatial_random_affine(xyz,
    scale  = (0.8,1.2),
    shear = (-0.15,0.15),
    shift  = (-0.1,0.1),
    degree = (-30,30),):
    
    center = torch.tensor([0,0])
    
    if scale is not None:
        scale = np.random.uniform(scale[0],scale[1])
        xyz = xyz * scale
        
    if shear is not None:
        #might not work when only osuing xy
        xy = xyz[:,:,:2]
        z = xyz[:,:,2:]
        shear_x = shear_y = np.random.uniform(shear[0],shear[1])
        if np.random.rand() < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.     
        shear_mat = torch.tensor([[1.,shear_x],
                                  [shear_y,1.]])    
        xy = xy @ shear_mat
        center = center + torch.tensor([shear_y, shear_x])
        xyz = torch.concat([xy,z], axis=-1)
        
    if degree is not None:
        xy = xyz[...,:2]
        z = xyz[...,2:]
        xy -= center
        degree = np.random.uniform(degree[0],degree[1])
        radian = degree/180*np.pi
        c = math.cos(radian)
        s = math.sin(radian)
        
        rotate_mat = torch.tensor([[c,s],
                                   [-s, c]])
        
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = torch.cat([xy,z], axis=-1)
        
    if shift is not None:
        shift = np.random.uniform(shift[0],shift[1])
        xyz = xyz + shift
        
    return xyz

def temporal_crop(x, length):
    l = x.shape[0]
    max_l = np.clip(l-length,1,length)
    offset = int(np.random.uniform(0, max_l))
    x = x[offset:offset+length]
    return x

def temporal_mask(x, size=(0.2,0.4), mask_value=float('nan')):
    l = x.shape[0]
    mask_size = np.random.uniform(size[0],size[1])
    mask_size = int(l * mask_size)
    max_mask = np.clip(l-mask_size,1,l)
    mask_offset = int(np.random.uniform(0, max_mask))
    x_new = x.contiguous()
    x_new[mask_offset:mask_offset+mask_size] = torch.tensor(mask_value)
    return x_new


def scale(data, factor=0.3, p=0.5):
    if np.random.random() > p:
        return data
    
    distort = np.random.random() < p
    scale_factor = np.random.uniform(1 - factor, 1 + factor)

    for k in ['x', 'y', 'z']:
        distort_factor = np.random.uniform(1 - factor, 1 + factor) if distort else 0
        data[k] *= (scale_factor + distort_factor)
        
    return data

def spatial_mask(x, size=(0.5,1.), mask_value=float('nan')):
    mask_offset_y = np.random.uniform(x[...,1].min().item(),x[...,1].max().item())
    mask_offset_x = np.random.uniform(x[...,0].min().item(),x[...,0].max().item())
    mask_size = np.random.uniform(size[0],size[1])
    mask_x = (mask_offset_x<x[...,0]) & (x[...,0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y<x[...,1]) & (x[...,1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x_new = x.contiguous()
    x_new = x * (1-mask[:,:,None].float()) #+ mask_value[:,:,None] * mask_value
    return x_new

def crop_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # Crop
        if mode == "start":
            data = data[:max_len]
        else:
            offset = np.abs(diff) // 2
            data = data[offset: offset + max_len]
        mask = torch.ones_like(data[:,0,0])
        return data, mask
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2]))
    mask = torch.ones_like(data[:,0,0])
    data = torch.cat([data, padding * coef])
    mask = torch.cat([mask, padding[:,0,0] * coef])
    
    
    
    return data, mask

def interpolate_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # Crop
        data = F.interpolate(data.permute(1,2,0),max_len).permute(2,0,1)
        mask = torch.ones_like(data[:,0,0])
        return data, mask
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2]))
    mask = torch.ones_like(data[:,0,0])
    data = torch.cat([data, padding * coef])
    mask = torch.cat([mask, padding[:,0,0] * coef])
    return data, mask

def inner_cutmix(data, phrase):
    
    if (len(phrase) > 1) & (data.shape[0]>1):
        cut_off = np.random.rand()
        cut_off_data = int(data.shape[0] * cut_off)
        cut_off_data = max(cut_off_data,1)
        cut_off_data = min(cut_off_data,data.shape[0]-1)

        cut_off_phrase = int(len(phrase) * cut_off)
        cut_off_phrase = max(cut_off_phrase,1)
        cut_off_phrase = min(cut_off_phrase,len(phrase)-1)

        new_data = torch.cat([data[cut_off_data:],data[:cut_off_data]])
        new_phrase = phrase[cut_off_phrase:] + phrase[:cut_off_phrase]
        return new_data, new_phrase
    
    else:
        return data, phrase

def outer_cutmix(data, phrase,score, data2, phrase2,score2):
    cut_off = np.random.rand()
    
    
    
    cut_off_phrase = np.clip(round(len(phrase) * cut_off),1,len(phrase)-1)
    cut_off_phrase2 = np.clip(round(len(phrase2) * cut_off),1,len(phrase2)-1)
    
    cut_off_data = np.clip(round(data.shape[0] * cut_off),1,data.shape[0]-1)
    cut_off_data2 = np.clip(round(data2.shape[0] * cut_off),1,data2.shape[0]-1)

    if np.random.rand() < 0.5:
        new_phrase = phrase2[cut_off_phrase2:] + phrase[:cut_off_phrase]
        new_data = torch.cat([data2[cut_off_data2:], data[:cut_off_data]])
        new_score = cut_off*score + (1-cut_off) * score2
    else:
        new_phrase = phrase[cut_off_phrase:] + phrase2[:cut_off_phrase2]
        new_data = torch.cat([data[cut_off_data:], data2[:cut_off_data2]])
        new_score = cut_off*score2 + (1-cut_off) * score
    return new_data, new_phrase, new_score

# def interpolate(data, size=50):
#     interpolated = {}
#     for k in data.keys():
#         mode = "linear" if isinstance(data[k], torch.FloatTensor) else "nearest"
#         interpolated[k] = torch.nn.functional.interpolate(
#             data[k].T.unsqueeze(0).float(), size, mode=mode
#         )[0].T.to(data[k].dtype)
#     return interpolated

'''
df = pd.read_csv(cfg.train_df).query('fold==0')

mode="train"
class self:
    1
self = CustomDataset(df, cfg, aug = cfg.train_aug, mode = 'train')
idx = 10
aug = cfg.train_aug

# self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'valid')
batch = [self.__getitem__(i) for i in range(0, 16)]
# batch = tr_collate_fn(batch)
# batch = batch_to_device(batch, 'cpu')

loader = DataLoader(self , batch_size=64, shuffle=False)
batch = next(iter(loader))

'''

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train", symmetry_fn=None):

        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.aug = aug
        
        if mode =='train':
            to_drop = self.df['seq_len'] < cfg.min_seq_len
            self.df = self.df[~to_drop].copy()
            print(f'new shape {self.df.shape[0]}, dropped {to_drop.sum()} sequences shorter than min_seq_len {cfg.min_seq_len}')
        if 'score' not in self.df.columns:
            print('no score in columns')
            self.df['score'] = 0.5
        self.df['score'] = self.df['score'].clip(0,1)
        #input stuff
        with open(cfg.data_folder + 'inference_args.json', "r") as f:
            columns = json.load(f)['selected_columns']
        
        self.xyz_landmarks = np.array(columns)
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[:len(self.xyz_landmarks)//3]])
        
         
        symmetry = pd.read_csv(symmetry_fn).set_index('id')
        flipped_landmarks = symmetry.loc[landmarks]['corresponding_id'].values
        self.flip_array = np.where(landmarks[:,None]==flipped_landmarks[None,:])[1]
        
        self.max_len = cfg.max_len
        
        self.processor = Preprocessing()
        self.resize_mode = cfg.resize_mode
        
        #target stuff
        self.max_phrase = cfg.max_phrase
        self.char_to_num, self.num_to_char, _ = self.cfg.tokenizer
        self.pad_token_id = self.char_to_num[self.cfg.pad_token]
        self.start_token_id = self.char_to_num[self.cfg.start_token]
        self.end_token_id = self.char_to_num[self.cfg.end_token]

        self.flip_aug = cfg.flip_aug
        self.inner_cutmix_aug = cfg.inner_cutmix_aug
        self.outer_cutmix_aug = cfg.outer_cutmix_aug

        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder
        idx = 0

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        file_id, sequence_id,phrase,score = row[['file_id','sequence_id','phrase','score']]
        
        data = self.load_one(file_id, sequence_id)
        seq_len = data.shape[0]
        data = torch.from_numpy(data)
        
#         print('data_raw',data.shape)
#         return data
        if self.mode == 'train':
            if self.aug[0]:
                #reshape
                #seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3
                data = data.reshape(data.shape[0],3,-1).permute(0,2,1)
                data = self.augment(data,stage=0)
                data = data.permute(0,2,1).reshape(data.shape[0],-1)
                #seq_len, n_landmarks, 3 -> seq_len, 3* n_landmarks
#                 print('data0',data.shape)
                #reshape
                
            data = self.processor(data)
            
            if np.random.rand() < self.flip_aug:
                data = flip(data, self.flip_array) 
                
            if self.aug[1]:
                data = self.augment(data,stage=1)            
#                 print('data1',data.shape)
            if np.random.rand() < self.outer_cutmix_aug:

                participant_id = row['participant_id']
                sequence_id = row['sequence_id']
                mask = (self.df['participant_id']==participant_id) & (self.df['sequence_id']!=sequence_id)
                
                if mask.sum() > 0:
                    row2 = self.df[mask].sample(1).iloc[0]
                    file_id2, sequence_id2,phrase2,score2 = row2[['file_id','sequence_id','phrase','score']]
                    data2 = self.load_one(file_id2, sequence_id2)
                    seq_len2 = data2.shape[0]
                    data2 = torch.from_numpy(data2)
#                     print('data2_raw',data2.shape)
                    if self.aug[0]:
                        data2 = data2.reshape(data2.shape[0],3,-1).permute(0,2,1)
                        data2 = self.augment(data2,stage=0)
                        data2 = data2.permute(0,2,1).reshape(data2.shape[0],-1)
#                         print('data20',data2.shape)
                    data2 = self.processor(data2)    
                    if self.aug[1]:
                        data2 = self.augment(data2,stage=1)  
#                         print('data21',data2.shape)
                        
#                     print(data.shape, data2.shape)
                    data, phrase, score = outer_cutmix(data, phrase,score, data2, phrase2, score2)
                
            
            if np.random.rand() < self.inner_cutmix_aug:
                data, phrase = inner_cutmix(data, phrase)
                
            if self.aug[2]:
                data = self.augment(data, stage=2)
#                 print('data2',data.shape)

        else:
            data = self.processor(data)

#         if self.max_len is not None:
        if self.resize_mode == "crop_or_pad":
            data, mask = crop_or_pad(data, max_len=self.max_len)
        elif self.resize_mode == "interpolate_or_pad":
            data, mask = interpolate_or_pad(data, max_len=self.max_len)
        else:
            raise NotImplementedError
            # data = interpolate(data, size=self.max_len)  

        
        token_ids, attention_mask = self.tokenize(phrase)
        
        feature_dict = {'input':data,
                        'input_mask':mask,
                        'token_ids':token_ids, 
                        'attention_mask':attention_mask,
                        'seq_len':torch.tensor(seq_len),
                        'score':torch.tensor(score),
                       
                       }
        return feature_dict
    
    def augment(self,x,stage=0):
        x_aug = self.aug[stage](image=x)['image']
        return x_aug
    
    def tokenize(self,phrase):
        phrase_ids = [self.char_to_num[char] for char in phrase]
        if len(phrase_ids) > self.max_phrase - 1:
            phrase_ids = phrase_ids[:self.max_phrase - 1]
        phrase_ids = phrase_ids + [self.end_token_id]
        attention_mask = [1] * len(phrase_ids)
        
        to_pad = self.max_phrase - len(phrase_ids)
        phrase_ids = phrase_ids + [self.pad_token_id] * to_pad
        attention_mask = attention_mask + [0] * to_pad
        return torch.tensor(phrase_ids).long(), torch.tensor(attention_mask).long()
        
    
    def setup_tokenizer(self):
        with open (self.cfg.character_to_prediction_index_fn, "r") as f:
            char_to_num = json.load(f)

        n= len(char_to_num)
        char_to_num[self.cfg.pad_token] = n
        char_to_num[self.cfg.start_token] = n+1
        char_to_num[self.cfg.end_token] = n+2
        num_to_char = {j:i for i,j in char_to_num.items()}
        return char_to_num, num_to_char

    def __len__(self):
        return len(self.df)

    def load_one(self, file_id, sequence_id):
        path = self.data_folder + f'{file_id}/{sequence_id}.npy'
        data = np.load(path) # seq_len, 3* nlandmarks
        return data
