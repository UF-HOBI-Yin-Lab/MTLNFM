import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from LoadData import IPD_MA_STL,Load_saved_data_STL
from Util import EarlyStopper,seed_torch,get_model,prob_to_label
from train_test_framework import train,test
from sklearn.metrics import roc_auc_score,precision_score,accuracy_score,f1_score
import numpy as np

def main(task_number,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    data = IPD_MA_STL(dataset_path,task_number)
    training_set, other_set = train_test_split(data,test_size=0.2,shuffle=True)
    validation_set,test_set = train_test_split(other_set,test_size=0.5,shuffle=True)
    # dataset_path = './data/'
    # training_set = Load_saved_data_STL(dataset_path+'training_set.npy',2)
    # validation_set = Load_saved_data_STL(dataset_path+'validation_set.npy',2)
    # test_set = Load_saved_data_STL(dataset_path+'test_set.npy',2)
    train_data_loader = DataLoader(training_set, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    model_log = open(f'{save_dir}/STL_{model_name}_model_log.txt','a') 

    model = get_model(model_name).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    early_stopper = EarlyStopper(num_trials=5, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        targets, predicts, logloss = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: total_auc:', roc_auc_score(targets,predicts))
        if not early_stopper.is_continuable(model, roc_auc_score(targets,predicts), logloss):
            print(f'validation: best auc: {early_stopper.best_accuracy} best loss: {early_stopper.best_logloss}')
            break
    original_model = torch.load(f'{save_dir}/{model_name}.pt')     
    targets, predicts, logloss= test(original_model, test_data_loader, device)
    print(f'test total auc: {roc_auc_score(targets,predicts)}')
    print(f'test precision:{precision_score(targets,prob_to_label(predicts))},accuracy:{accuracy_score(targets,prob_to_label(predicts))},f1_score:{f1_score(targets,prob_to_label(predicts))}')
    print(f'test total loss: {logloss}')
    model_log.flush()


if __name__ == '__main__':
    import argparse
    seed_torch(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_number', default=2) #task id
    parser.add_argument('--dataset_path', default='./data/cutoff_75%_los10_training_data.npy')
    parser.add_argument('--model_name', default='stlnfm') 
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.001) 
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    main(args.task_number,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
