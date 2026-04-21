import csv
import os
from config import BASE_MODELS_DIR

log_dir  = os.path.join(BASE_MODELS_DIR, 'logs')
log_file = os.path.join(log_dir, 'logs.csv')

def csv_creator(filename, fields, rows):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerow(rows)


def logger(model_name, num_params, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, best_epoch, epochs, macro_f1score, desc):
    os.makedirs(log_dir, exist_ok=True)
    fields = ['Model', 'Number Of Params', 'Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Test Accuracy', 'Test Loss', 'Best Epoch', 'Epochs', 'Macro-F1 score', 'Description']
    print(f"The file will be saved here: {log_file}")
    row = [model_name, num_params, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, best_epoch, epochs, macro_f1score, desc]
    if not os.path.exists(log_file):
        csv_creator(log_file, fields, row)
    else:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)