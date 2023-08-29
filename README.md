## Google - ASL Fingerspelling Recognition with [neptune.ai](neptune.ai)

Competiton website [link](https://www.kaggle.com/competitions/asl-fingerspelling).  

[<img src="https://img.youtube.com/vi/q3xKB3dfvtA/hqdefault.jpg" width="600" height="500"
/>](https://www.youtube.com/embed/q3xKB3dfvtA)

## Getting started

Install the packages with `pip install -r requirements.txt`.

Check the competition website to dowload the data and place it in a folder called `datamount/`.

Alternatively you can download a pre-processed dataset using the [kaggle api](https://github.com/Kaggle/kaggle-api) and 
place it in a folder `datamount/train_landmarks_v3/`.
```
kaggle datasets download -d darraghdog/processed-train-landmarks-v03
```
If you would like to process the data yourself you can use,
```
python scripts/convert2numpy_v3.py --input_dir <DOWNLOADED_KAGGLE_PARQUET_FOLDER> \
        --train_df datamount/train.csv \
        ----output_dir datamount/train_landmarks_v3/
```
  
Check out the neptune.ai [quickstart](https://docs.neptune.ai/usage/quickstart/) to set up your neptune project 
and place the your projects api token in a `.env` file, like below.
```
$ cat .env 
NEPTUNE_PROJECT_NAME=light/asl
NEPTUNE_ASL_TOKEN=eyJhcGlfYWRkcmV....
```

## Start training a config
```
python train.py -C cfg_dh_61g3
```
