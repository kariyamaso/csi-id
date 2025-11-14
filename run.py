import argparse
import datetime as dt
import os
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from util import load_data_n_model


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
    return


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return

    
def parse_args():
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT','SSM','Mamba'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a pretrained state_dict to load before training.')
    parser.add_argument('--eval-only', action='store_true', help='Skip training and only run evaluation.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory to store logs (mirrors train_all format).')
    parser.add_argument('--log-file', type=str, default=None, help='Explicit log file path. Overrides --log-dir if both are provided.')
    return parser.parse_args()


def run_experiment(args):
    root = './Data/' 
    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint}")

    if args.eval_only:
        test(
            model=model,
            tensor_loader=test_loader,
            criterion=criterion,
            device=device
        )
        return

    train(
        model=model,
        tensor_loader= train_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
         )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )


def main():
    args = parse_args()
    log_path = None
    if args.log_file:
        log_path = Path(args.log_file)
    elif args.log_dir:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = Path(args.log_dir) / args.dataset / f"{timestamp}_{args.model}.log"

    if not log_path:
        run_experiment(args)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        log_file = stack.enter_context(log_path.open("w", buffering=1))
        cmdline = f"python {' '.join(sys.argv[1:])}"
        log_file.write(f"$ {cmdline}\n\n")
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            run_experiment(args)
    print(f"Logs written to {log_path}")


if __name__ == "__main__":
    main()
