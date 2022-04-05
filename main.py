from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
import os
from sklearn import metrics
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import glob


normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)
normal2_test_dataset = Normal_Loader(is_train=2)

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)
anomaly2_test_dataset = Anomaly_Loader(is_train=2)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
normal2_test_loader = DataLoader(normal2_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
anomaly2_test_loader = DataLoader(anomaly2_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Learner(input_dim=2048, drop_p=0.0).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr= 0.001, weight_decay=0.0010000000474974513)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL

def train(epoch): 
    
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)): 
        # print(batch_idx)
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1) #torch.cat là hàm nối chuỗi anomaly_inputs và normal_inputs
        batch_size = inputs.shape[0] # inputs.shape  = torch.Size([30, 64, 2048])    và batch_size = 30
        
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)   

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
    print('loss = {}', train_loss/len(normal_train_loader))
    scheduler.step()

def test_abnormal(epoch):
    model.eval()
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            
            # anomaly
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            #print(score) # có 32 ô
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]
            # print(score_list)
            gt_list = np.zeros(frames[0])
            # print(frames)
            # print(len(gts)//2)
            for k in range(len(gts)//2):
                s = gts[k*2]
                # print(s)
                e = min(gts[k*2+1], frames) # why do that???
                # print(e)
                
                gt_list[s-1:e] = 1
            # print(gt_list)
            #normal
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0]//16, 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])

            ################################
            score_list4 = np.concatenate((score_list, score_list2), axis=0)
            gt_list4 = np.concatenate((gt_list, gt_list2), axis=0)

            # fpr, tpr, thresholds = metrics.roc_curve(gt_list4, score_list4, pos_label=1)
            fpr, tpr, thresholds = metrics.roc_curve(gt_list4, score_list4, pos_label=1)
            # print(fpr)
            auc += metrics.auc(fpr, tpr)
        print('auc = ', auc/5)
        num = auc/5
        # print('num = ', num)
        if((num) > 0.64 and (num) < 0.66):
            print("arrest");
        # print('score = ', score)
        # print('tpr = ', tpr)
        
FILE ="model.pth"
# path= "Dataset/workspace/DATA/UCF-Crime/"



# for i, (data) in enumerate(zip(anomaly2_test_loader)):
#     print(i)
#     # print(data)
    
#     test_abnormal(data)

# for epoch in range(0, 1):
#     train(epoch)
#     test_abnormal(epoch)



# # #train save load model 
# for epoch in range(0, 200):
#     train(epoch)
#     # torch.save(model.state_dict(), FILE)
#     # test_abnormal(epoch)
# print("trained")
# torch.save(model.state_dict(), FILE)
# print("saved")

model.load_state_dict(torch.load(FILE))
model.eval()
print("loaded")

# for i, (data) in enumerate(zip(normal2_test_loader)):
#     print(i)
#     # print(data)
#     test_abnormal(data)

for epoch in range(0, 10):
    test_abnormal(epoch)












# for epoch in range(0, 100):
#     print(epoch)
#     test_abnormal(epoch)
# print("printed epoch")


##### test video #########


# for i, (data) in enumerate(zip(normal2_test_loader)):
#     print(i)
#     # print(data)
#     test_abnormal(data)
    
# for param in model.parameters():
#     print(param)
#     test_abnormal(param)

