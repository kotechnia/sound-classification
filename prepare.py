from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json
import argparse

def get_key(path):
    path = os.path.basename(path)
    path = path.split('_')
    path = path[:-1]
    path = "_".join(path)
    return path

def json_load(path):
    with open(path) as f:
        return json.load(f)
    
def train_val_test_fold(value):
    if value < 8:
        return 'train'
    elif value < 9:
        return 'val'
    else:
        return 'test'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        default='./data'
    )
    parser.add_argument(
        "--dataset_path",
        default='./dataset.csv'
    )
    args = parser.parse_args()

    wavs_sn = glob(os.path.join(args.dataroot,'**/*_SN.wav'), recursive=True)
    wavs_ns = glob(os.path.join(args.dataroot,'**/*_NS.wav'), recursive=True)
    jsons = glob(os.path.join(args.dataroot, '**/*.json'), recursive=True)

    wavs_sn = pd.DataFrame({"sn_file_name":wavs_sn})
    wavs_sn['key'] = wavs_sn['sn_file_name'].map(get_key)
    wavs_ns = pd.DataFrame({"ns_file_name":wavs_ns})
    wavs_ns['key'] = wavs_ns['ns_file_name'].map(get_key)

    jsons = pd.DataFrame({"script_path":jsons})
    jsons['key'] = jsons['script_path'].map(get_key)

    df = pd.merge(wavs_ns, wavs_sn, on=['key'])
    df = pd.merge(df, jsons, on=['key'])

    for i in range(len(df)):
        data = json_load(df.loc[i, 'script_path'])
        df.loc[i, 'category'] = data['soundInfo']['sdCategory']
        df.loc[i, 'subcategory'] = data['soundInfo']['sdsubCategory']

    categories = df.groupby(by=['category', 'subcategory']).count().reset_index()
    class_to_index = {}
    for i in range(len(categories)):
        category = categories.loc[i, 'category']
        subcategory = categories.loc[i, 'subcategory']
        class_to_index[f'{category}_{subcategory}'] = i

    train_val_test = np.arange(df.shape[0])
    np.random.shuffle(train_val_test)
    train_val_test = train_val_test%10
    train_val_test = list(map(train_val_test_fold, train_val_test))
    df['fold'] = train_val_test

    df_meta=[]
    for i in tqdm(range(len(df)), ascii=True):
        script_path = df.loc[i, 'script_path']
        key = df.loc[i, 'key']

        _category = df.loc[i, 'category']
        _subcategory = df.loc[i, 'subcategory']
        fold = df.loc[i, 'fold']

        category = f"{_category}_{_subcategory}"
        target = class_to_index[category]

        for column in ['sn_file_name', 'ns_file_name']:
            filename = df.loc[i, column]
            src_file = filename

            src_path = os.path.join(filename)

            df_meta.append({
                'filename':src_path,
                'fold':fold,
                'target':target,
                'category':category,
                'key':key,
                'detail':script_path,
            })

    df_meta = pd.DataFrame(df_meta)
    df_meta.to_csv(args.dataset_path, index=False)

    print(df_meta.groupby(by=['fold','category']).count())


if __name__ == '__main__':
    main()
