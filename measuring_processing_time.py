import torch, os, config
import torch.nn as nn
import pandas as pd
import torchvision.models as models
import time

import pandas as pd, sys
import time

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

def measureProcessingTime(x, model, start_count=0):
  processing_time_dict = {}
  count_acc = 0
  for i, exitBlock in enumerate(model.exits):
    x, processing_time, count = getProcessingTime(x, model.stages[i], start_count = count_acc)
    processing_time_dict.update(processing_time)
    count_acc+=count
    
    start = time.time()
    pred = exitBlock(x)
    end = time.time()
    processing_time = end-start
    processing_time_dict.update({"b_%s"%(i+1): processing_time})

  x, processing_time, count = getProcessingTime(x, model.stages[-1], start_count = count_acc)
  processing_time_dict.update(processing_time)
  count_acc+=count
  x = x.view(x.size(0), -1)

  x, processing_time, count = getProcessingTime(x, model.fully_connected, start_count = count_acc)
  processing_time_dict.update(processing_time)
  return processing_time_dict


def measuring_processing_time_b_alexnet(x, model, savePath, use_gpu, n_rounds=50):
  df = pd.DataFrame(columns=[])
  use_gpu_str = "gpu" if use_gpu else "cpu"
  for n in range(1, n_rounds+1):
    print("Round: %s"%(n))
    processing_time_dict = measureProcessingTime(x, model)
    df = df.append(pd.Series(processing_time_dict), ignore_index=True)

  df.to_csv("./processing_time_b_alexnet_%s.csv"%(use_gpu_str))

x = torch.rand(1, 3, 224, 224)
model = models.alexnet()
savePath = os.path.join(config.DIR_NAME, "appEdge", "api", "processing_time", "processing_time_b_alexnet_cpu_edge_notebook.csv")
saveProcessingTimeAlexNet(x, model, savePath)