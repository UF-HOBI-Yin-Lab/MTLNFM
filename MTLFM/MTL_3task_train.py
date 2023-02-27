import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from LoadData import IPD_MA_MTL,Load_saved_data_MTL
from Util import EarlyStopper,get_model,prob_to_label,seed_torch
from train_test_framework import train_3task,test_3task
from sklearn.metrics import roc_auc_score,precision_score,f1_score,accuracy_score
import numpy as np
def main(dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    data = IPD_MA_MTL(dataset_path)
    training_set, other_set = train_test_split(data,test_size=0.2,shuffle=True)
    validation_set,test_set = train_test_split(other_set,test_size=0.5,shuffle=True)
#     dataset_path = './data/'
#     training_set = Load_saved_data_MTL(dataset_path+'training_set.npy')
#     validation_set = Load_saved_data_MTL(dataset_path+'validation_set.npy')
#     test_set = Load_saved_data_MTL(dataset_path+'test_set.npy')

    train_data_loader = DataLoader(training_set, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    model_log = open(f'{save_dir}/3task_model_log.txt','a') 

    model = get_model(model_name).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    early_stopper = EarlyStopper(num_trials=5, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
        train_3task(model, optimizer, train_data_loader, criterion, device)
        total_target,total_predict,loss = test_3task(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: total_auc:', roc_auc_score(total_target[0], total_predict[0])
              , 'task1_auc:',roc_auc_score(total_target[1], total_predict[1])
              , 'task2_auc:',roc_auc_score(total_target[2], total_predict[2])
              , 'task3_auc:',roc_auc_score(total_target[3], total_predict[3]))
        print('epoch:', epoch_i, 'validation: total_logloss:', loss[0], 'task1:', loss[1],'task2:', loss[2],'task3:', loss[3])
        if not early_stopper.is_continuable(model,roc_auc_score(total_target[0], total_predict[0]), loss[0]):
            print(f'validation: best auc: {early_stopper.best_accuracy} best loss: {early_stopper.best_logloss}')
            break

    original_model = torch.load(f'{save_dir}/{model_name}.pt')     
    total_target, total_predict, logloss= test_3task(original_model, test_data_loader, device)
    print(f'test total auc: {roc_auc_score(total_target[0], total_predict[0])},task1 auc: {roc_auc_score(total_target[1], total_predict[1])},task2 auc: {roc_auc_score(total_target[2], total_predict[2])},task3 auc: {roc_auc_score(total_target[3], total_predict[3])}')
    print('test total precision:', precision_score(total_target[0], prob_to_label(total_predict[0]))
            , 'task1_precision:',precision_score(total_target[1], prob_to_label(total_predict[1]))
            , 'task2_precision:',precision_score(total_target[2], prob_to_label(total_predict[2]))
            , 'task3_precision:',precision_score(total_target[3], prob_to_label(total_predict[3])))
    print('test totall accuracy', accuracy_score(total_target[0], prob_to_label(total_predict[0]))
            , 'task1_accuracy:',accuracy_score(total_target[1], prob_to_label(total_predict[1]))
            , 'task2_accuracy:',accuracy_score(total_target[2], prob_to_label(total_predict[2]))
            , 'task3_accuracy:',accuracy_score(total_target[3], prob_to_label(total_predict[3])))
    print('test totall f1', f1_score(total_target[0], prob_to_label(total_predict[0]))
            , 'task1_f1:',f1_score(total_target[1], prob_to_label(total_predict[1]))
            , 'task2_f1:',f1_score(total_target[2], prob_to_label(total_predict[2]))
            , 'task3_f1:',f1_score(total_target[3], prob_to_label(total_predict[3])))
    print(f'test total loss: {logloss[0]},task1 loss: {logloss[1]},task2 loss: {logloss[2]},task3 loss: {logloss[3]}')
    model_log.flush()




if __name__ == '__main__':
    import argparse

    seed_torch(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data/cutoff_75%_los10_training_data.npy')
    parser.add_argument('--model_name', default='mtlnfm')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    main(args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
