from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import argparse
import json

parser = argparse.ArgumentParser("convert original train to numpy files")
arg = parser.add_argument
arg('--input_dir', type=str, default="/raid/asl-fingerspelling/train_landmarks/", help='original input folder holding large parquet files')
arg('--output_dir', type=str, default="/raid/asl-fingerspelling/train_landmarks_v3/", help='output folder holding /*/*.npy')
arg('--n_cores', type=int, default=32)
arg('--train_df', type=str, default="/mount/asl/data/train.csv")
args = parser.parse_args()

print(f'Args : {args}')

'''
args.input_dir = 'datamount/train_landmarks/'
args.output_dir = 'datamount/train_landmarks_v3/'
args.train_df = 'datamount/train.csv'

'''


train = pd.read_csv(args.train_df)

# train_cols in right order
all_cols = [f'face_{i}' for i in range(468)] 
all_cols += [f'left_hand_{i}' for i in range(21)] 
all_cols += [f'pose_{i}' for i in range(33)]
all_cols += [f'right_hand_{i}' for i in range(21)]
all_cols = np.array(all_cols)


#1st place kept landmarks

NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
LIP = [ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

LARMS = [501, 503, 505, 507, 509, 511]
RARMS = [500, 502, 504, 506, 508, 510]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE + LARMS + RARMS

kept_cols = all_cols[POINT_LANDMARKS]
n_landmarks = len(kept_cols)

kept_cols_xyz = np.array(['x_' + c for c in kept_cols] + ['y_' + c for c in kept_cols] + ['z_' + c for c in kept_cols])


TARGET_FOLDER = args.output_dir

file_ids = train['file_id'].unique()


def do_one(file_id):
    os.makedirs(TARGET_FOLDER + f'{file_id}/', exist_ok=True)
    df = pd.read_parquet(f'{args.input_dir}{file_id}.parquet').reset_index()
    sequence_ids = df['sequence_id'].unique()
    for sequence_id in sequence_ids:
        df_seq = df[df['sequence_id']==sequence_id].copy()
        vals = df_seq[kept_cols_xyz].values
        np.save(TARGET_FOLDER + f'{file_id}/{sequence_id}.npy',vals)

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    with mp.Pool(args.n_cores) as p:
        res = list(tqdm(p.imap(do_one,file_ids), total=len(file_ids)))
        
    selected_columns_dict = {"selected_columns": kept_cols_xyz.tolist()}
    
    with open(f'{TARGET_FOLDER}inference_args.json', "w") as f:
        json.dump(selected_columns_dict, f)
        
    
    np.save(TARGET_FOLDER + 'columns.npy',kept_cols_xyz)

