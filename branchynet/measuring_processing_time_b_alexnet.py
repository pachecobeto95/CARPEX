import pandas as pd, sys
import time
from branchyNet import BranchyNet
import torch, os

def getProcessingTime(x, model, start_count = 0):
  count = 0
  processing_time_dict = {}
  for layer in model:
    start = time.time()
    x = layer(x)
    end = time.time()
    processing_time = end-start
    count +=1
    processing_time_dict.update({"l_%s"%(count+start_count): processing_time})

  return x, processing_time_dict, count      

def measureProcessingTime(x, model, branches_positions, start_count=0):
  processing_time_dict = {}
  count_acc = 0
  for i, (exitBlock, b_pos) in enumerate(zip(model.exits, branches_positions)):
    x, processing_time, count = getProcessingTime(x, model.stages[i], start_count = count_acc)
    processing_time_dict.update(processing_time)
    count_acc+=count
    
    start = time.time()
    pred = exitBlock(x)
    end = time.time()
    processing_time = end-start
    processing_time_dict.update({"b_%s"%(b_pos): processing_time})

  x, processing_time, count = getProcessingTime(x, model.stages[-1], start_count = count_acc)
  processing_time_dict.update(processing_time)
  count_acc+=count
  x = x.view(x.size(0), -1)

  x, processing_time, count = getProcessingTime(x, model.fully_connected, start_count = count_acc)
  processing_time_dict.update(processing_time)
  return processing_time_dict


def measuring_processing_time_b_alexnet(x, model, savePath, use_gpu, branches_positions, n_rounds=50):
  df = pd.DataFrame(columns=[])
  use_gpu_str = "gpu" if use_gpu else "cpu"
  for n in range(1, n_rounds+1):
    print("Round: %s"%(n))
    processing_time_dict = measureProcessingTime(x, model, branches_positions)
    df = df.append(pd.Series(processing_time_dict), ignore_index=True)

  df.to_csv("processing_time_b_alexnet_%s_notebook.csv"%(use_gpu_str))

feature_extraction = False
img_dim = 224
n_classes = 10
use_gpu = False
pretrained = True
imageNet = False
#device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu)else 'cpu')
device = "cpu"
model_name = "AlexNet"
dataset_name = "cifar"
is_adaptive_learning_rate = True
branches_positions = [2, 5, 7]
n_branches = len(branches_positions)
savePath = "processing_time_b_alexnet.csv"

model = BranchyNet(model_name, dataset_name, n_classes, pretrained, imageNet, feature_extraction, n_branches,
                   img_dim, exit_type=None, branches_positions=branches_positions)
model.loadEarlyExitModel()
model = model.to(device)
x = torch.rand(1, 3, 224, 224).to(device)
measuring_processing_time_b_alexnet(x, model, savePath, use_gpu, branches_positions)