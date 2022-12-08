import librosa
import os
import torch
from torch import nn
from torch.nn import Softmax
from model.esresnet import ESResNetAttention, ESResNet
from utils.datasets import ESC50, NIA2022
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import re
from datetime import datetime
from glob import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='tscn helper.')
    parser.add_argument(
        "--model",
        default='./saved_models/NIA2022_STFT_ESRN-CV1',
    )
    parser.add_argument(
        "--results_path",
        default='./results.csv',
    )

    args = parser.parse_args()
    model_root = os.path.join(args.model, "*.pth")
    model_list = glob(model_root)
    model_list = sorted(model_list)
    model_path = model_list[-1]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = '.'
    meta = pd.read_csv(os.path.join('dataset.csv'))
    meta = meta[meta['fold'] == 'test'].reset_index(drop=True)

    model = ESResNet(n_fft=2048, hop_length=561, win_length=1654,
            window="blackmanharris", normalized=True,
            onesided=True, spec_height=-1, spec_width=-1, 
            num_classes=8, pretrained=True, lock_pretrained=False)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    for i in tqdm(range(len(meta))):
        filename = meta.loc[i, 'filename']
        category = meta.loc[i, 'target']

        _, sample_rate, y = NIA2022._load_worker(i, os.path.join(data_dir, filename), 44100)
        y = y.reshape(1, 1, -1)
        y = torch.Tensor(y)

        result = model(y)
        result = result.reshape(-1)

        pred = int(torch.argmax(result).cpu())
    
        results.append({
            'filename':filename,
            'predict':pred,
            'label':category,
        })

    results = pd.DataFrame(results)
    results.to_csv(args.results_path, index=False)
    f1_score_results = f1_score(results['label'].tolist(), results['predict'].tolist(), average='micro')
    print(f"{datetime.now().strftime('%Y%m%d-%H%M%S')} f1_score : {f1_score_results}")

if __name__ == '__main__':

    main()
