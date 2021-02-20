import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(description='Summarize log files.')
parser.add_argument('logs_dir', help='Path to log files folder containing .txt files')
args = parser.parse_args()

model_best_val_epoch = {}
for file in os.listdir(args.logs_dir):
    if os.path.splitext(file)[1].lower() != '.txt':
        continue
    path = os.path.join(args.logs_dir, file)
    model_name = os.path.splitext(file)[0]
    with open(path, 'r') as file_obj:
        header_line = file_obj.readline()
        best_val_epoch = 0
        best_epoch_val_loss = np.inf
        while True:
            line = file_obj.readline()
            if len(line) < 1 or line is None:
                break
            columns = line.split(',')
            columns = list(map(float, columns))
            epoch,train_loss,train_perplexity,val_loss,val_perplexity = columns
            if val_loss < best_epoch_val_loss:
                best_epoch_val_loss = val_loss
                best_val_epoch = epoch
        if best_epoch_val_loss != np.inf:
            model_best_val_epoch[model_name] = { 'epoch' : best_val_epoch, 'val_loss' : best_epoch_val_loss }

models_sorted_by_val_loss = sorted(model_best_val_epoch.keys(), key=lambda k: model_best_val_epoch[k]['val_loss'])
print('Model Name\tValidation Loss\tEpoch')
for model_name in models_sorted_by_val_loss:
    best_epoch = model_best_val_epoch[model_name]
    val_loss = best_epoch['val_loss']
    epoch_num = best_epoch['epoch']
    print(f'{model_name}\t{val_loss}\t{epoch_num}')
