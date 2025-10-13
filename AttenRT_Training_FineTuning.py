import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8) # for reproduce

import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
#then import my own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight

from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns; sns.set()
from IPython.display import SVG, display

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


task_name = 'SMRT'
tasks = ['SMRT']

random_seed = 101 # 69，103, 107
# 101
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
batch_size = 32
epochs = 800
p_dropout= 0.1
fingerprint_dim = 300
weight_decay = 4.5 # also known as l2_regularization_lambda
learning_rate = 3.5
radius = 2
T = 2
per_task_output_units_num = 1 # for regression model
output_units_num = len(tasks) * per_task_output_units_num

raw_filename = "./data/SHJTbas.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ",len(smilesList))
atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []
for smiles in smilesList:
    try:        
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        print(smiles)
        pass
print("number of successfully processed smiles: ", len(remained_smiles))
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
# print(smiles_tasks_df)
smiles_tasks_df['cano_smiles'] =canonical_smiles_list
assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)
# feature_dicts = get_smiles_dicts(smilesList)
remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)

test_df = remained_df.sample(frac=0.2,random_state=random_seed)
test_df.to_csv('./data/SHJTbas_test.csv', sep=',', index=False)
train_df = remained_df.drop(test_df.index)
train_df.to_csv('./data/SHJTbas_train.csv', sep=',', index=False)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]
loss_function = nn.MSELoss()
model = Fingerprint(radius, T, num_atom_features, num_bond_features,
            fingerprint_dim, output_units_num, p_dropout)
model.cuda()

best_model_name = 'model_SMRT_Mon_Nov_18_13-27-27_2024_199.pt'
model = torch.load('saved_models/best_models/'+ best_model_name)

# optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
# optimizer = optim.SGD(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)

# tensorboard = SummaryWriter(log_dir="runs/"+start_time+"_"+prefix_filename+"_"+str(fingerprint_dim)+"_"+str(p_dropout))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
        
def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    #shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch,:]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df[tasks[0]].values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        
        model.zero_grad()
        loss = loss_function(mol_prediction, torch.Tensor(y_val).view(-1,1))     
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
def evalu(model, dataset):
    model.eval()
    test_MAE_list = []
    test_MSE_list = []
    y_pred_list = []
    y_label_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch) 
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch,:]
        smiles_list = batch_df.cano_smiles.values
#         print(batch_df)
        y_val = batch_df[tasks[0]].values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')        
        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
#         print(x_mask[:2],atoms_prediction.shape, mol_prediction,MSE)
        
        test_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
        test_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
        y_pred_list.extend(mol_prediction.data.squeeze().cpu().numpy())
        y_label_list.extend(torch.Tensor(y_val).view(-1,1).data.squeeze().cpu().numpy())
    return np.array(test_MAE_list).mean(), np.array(test_MSE_list).mean(),y_pred_list,y_label_list


# 外部测试集预测
def predict_external(model, dataset):
    model.eval()
    y_pred_list = []
    y_label_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch,:]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df[tasks[0]].values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        _, mol_prediction = model(torch.Tensor(x_atom), torch.Tensor(x_bonds), 
                                  torch.cuda.LongTensor(x_atom_index), torch.cuda.LongTensor(x_bond_index), 
                                  torch.Tensor(x_mask))
        
        y_pred_list.extend(mol_prediction.data.squeeze().cpu().numpy())
        y_label_list.extend(torch.Tensor(y_val).view(-1,1).data.squeeze().cpu().numpy())
    return y_pred_list, y_label_list


best_param ={}
best_param["train_epoch"] = 0
best_param["test_epoch"] = 0
best_param["train_MSE"] = 9e8
best_param["test_MSE"] = 9e8

time_value = start_time.split('_')
logs_names = '_'.join([str(i) for i in [time_value[-1],time_value[1],time_value[2],batch_size,str(p_dropout).split('.')[-1],fingerprint_dim,radius,T,weight_decay,learning_rate]])

with open('./logs/' + logs_names+ '_train_detail' + '.logs','w' ) as wpklf: 
    re = ['epoch','loss','train_MAE','train_MSE','train_RMSE','train_r2','test_MAE','test_MSE','test_RMSE','test_r2']
    wpklf.write('\t'.join(re) + '\n')
    for epoch in range(epochs):
        # Training model
        loss = train(model, train_df, optimizer, loss_function)
        # Evaluate the model
        train_MAE, train_MSE,train_pred,train_label = evalu(model, train_df)
        train_RMSE = sqrt(mean_squared_error(train_label, train_pred))
        train_r2 = r2_score(train_label, train_pred)
        
        test_MAE, test_MSE,test_pred,test_label = evalu(model, test_df)
        test_RMSE = sqrt(mean_squared_error(test_label, test_pred))
        test_r2 = r2_score(test_label, test_pred)

        if train_MSE < best_param["train_MSE"]:
            best_param["train_epoch"] = epoch
            best_param["train_MSE"] = train_MSE
        if test_MSE < best_param["test_MSE"]:
            best_param["test_epoch"] = epoch
            best_param["test_MSE"] = test_MSE
            if test_MAE < 10:
                 torch.save(model, 'saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(epoch)+'.pt')
        if (epoch - best_param["train_epoch"] >20) and (epoch - best_param["test_epoch"] >28):        
            break
        
        re = [epoch,loss,train_MAE,train_MSE,train_RMSE,train_r2,test_MAE,test_MSE,test_RMSE,test_r2]
        print('\t'.join([str(ms) for ms in re]))
        
        wpklf.write('\t'.join([str(ms) for ms in re]) + '\n')
        
        
        
# evaluate model
best_model = torch.load('saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(best_param["test_epoch"])+'.pt') 
print("Best Model:  " , 'saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(best_param["test_epoch"])+'.pt')

best_model_dict = best_model.state_dict()
best_model_wts = copy.deepcopy(best_model_dict)

model.load_state_dict(best_model_wts)
(best_model.align[0].weight == model.align[0].weight).all()
test_MAE, test_MSE,test_pred,test_label = evalu(model, test_df)
print("best epoch:",best_param["test_epoch"],"\n","test MSE:",test_MSE,'\n',"test RMSE:",np.sqrt(test_MSE))

print('Internal Test Set:')
print('MAE; ', mean_absolute_error(test_label, test_pred))
print('MSE; ', mean_squared_error(test_label, test_pred))
print('RMSE; ', sqrt(mean_squared_error(test_label, test_pred)))
print('R2; ', r2_score(test_label, test_pred))


