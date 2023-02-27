import torch
import tqdm
import numpy as np

def get_task_target_2task(target):
    target1 = target[:,0]
    target2 = target[:,1]
    return target1,target2

def get_task_target_3task(target):
    target1 = target[:,0]
    target2 = target[:,1]
    target3 = target[:,2]
    return target1,target2,target3

def train(model, optimizer, data_loader, criterion, device,log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (field,target,mask,mask_value) in enumerate(tk0):
        field = field.type(torch.LongTensor).to(device)
        target,mask,mask_value = target.to(device),mask.to(device),mask_value.to(device)
        y = model(field,mask,mask_value)
        loss = criterion(y, target.float()) 
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    total_loss = list()
    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for i,(field,target,mask,mask_value) in enumerate(tk0):
            field = field.type(torch.LongTensor).to(device)
            target,mask,mask_value = target.to(device),mask.to(device),mask_value.to(device)
            y = model(field,mask,mask_value)
            criterion = torch.nn.BCELoss()
            loss = criterion(y, target.float())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            total_loss.append(loss.tolist())
    return targets,predicts,np.mean(total_loss)

def train_2task(model, optimizer, data_loader, criterion, device,log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (field,target,mask,mask_value) in enumerate(tk0):
        field = field.type(torch.LongTensor).to(device)
        target,mask,mask_value = target.to(device),mask.to(device),mask_value.to(device)
        y1,y2 = model(field,mask,mask_value)
        target1,target2 = get_task_target_2task(target)
        
        loss1 = criterion(y1, target1.float())
        loss2 = criterion(y2, target2.float())

        loss = loss1*loss1 + loss2*loss2
                
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test_2task(model, data_loader, device):
    model.eval()
    total_target = [[] for i in range(4)]
    total_loss = [[] for i in range(4)]
    total_predict = [[] for i in range(4)]
    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for i, (field,target,mask,mask_value) in enumerate(tk0):
            field = field.type(torch.LongTensor).to(device)
            target,mask,mask_value = target.to(device),mask.to(device),mask_value.to(device)
            y1,y2 = model(field,mask,mask_value)
            target1,target2 = get_task_target_2task(target)
            criterion= torch.nn.BCELoss()
        
            loss1 = criterion(y1, target1.float())
            loss2 = criterion(y2, target2.float())
            
            loss = loss1*loss1 + loss2*loss2 
            
            total_target[0].extend(target1.tolist())
            total_target[0].extend(target2.tolist())
            total_target[1].extend(target1.tolist())
            total_target[2].extend(target2.tolist())
            total_loss[0].append(loss.tolist())
            total_loss[1].append(loss1.tolist())
            total_loss[2].append(loss2.tolist())
            total_predict[0].extend(y1.tolist())
            total_predict[0].extend(y2.tolist())
            total_predict[1].extend(y1.tolist())
            total_predict[2].extend(y2.tolist())
            
    return  total_target,total_predict,[np.mean(total_loss[0]),
            np.mean(total_loss[1]),
            np.mean(total_loss[2])]

def train_3task(model, optimizer, data_loader, criterion,device,log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (field,target,mask,mask_value) in enumerate(tk0):
        field = field.type(torch.LongTensor).to(device)
        target,mask,mask_value = target.to(device),mask.to(device),mask_value.to(device)
        y1,y2,y3 = model(field,mask,mask_value)
        target1,target2,target3 = get_task_target_3task(target)

        loss1 = criterion(y1, target1.float())
        loss2 = criterion(y2, target2.float())
        loss3 = criterion(y3, target3.float())
        
        loss = loss1*loss1 + loss2*loss2 + loss3*loss3
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
            
def test_3task(model, data_loader, device):
    model.eval()
    total_target = [[] for i in range(4)]
    total_loss = [[] for i in range(4)]
    total_predict = [[] for i in range(4)]
    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for i, (field,target,mask,mask_value) in enumerate(tk0):
            field = field.type(torch.LongTensor).to(device)
            target,mask,mask_value = target.to(device),mask.to(device),mask_value.to(device)
            y1,y2,y3 = model(field,mask,mask_value)
            target1,target2,target3 = get_task_target_3task(target)
            criterion= torch.nn.BCELoss()
        
            loss1 = criterion(y1, target1.float())
            loss2 = criterion(y2, target2.float())
            loss3 = criterion(y3, target3.float())
            loss = loss1*loss1 + loss2*loss2 + loss3*loss3
            
            total_target[0].extend(target1.tolist())
            total_target[0].extend(target2.tolist())
            total_target[0].extend(target3.tolist())
            total_target[1].extend(target1.tolist())
            total_target[2].extend(target2.tolist())
            total_target[3].extend(target3.tolist())
            total_loss[0].append(loss.tolist())
            total_loss[1].append(loss1.tolist())
            total_loss[2].append(loss2.tolist())
            total_loss[3].append(loss3.tolist())
            total_predict[0].extend(y1.tolist())
            total_predict[0].extend(y2.tolist())
            total_predict[0].extend(y3.tolist())
            total_predict[1].extend(y1.tolist())
            total_predict[2].extend(y2.tolist())
            total_predict[3].extend(y3.tolist())
            
    return  total_target,total_predict,[np.mean(total_loss[0]),
            np.mean(total_loss[1]),
            np.mean(total_loss[2]),   
            np.mean(total_loss[3])]


