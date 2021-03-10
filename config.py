import os

DIR_NAME = os.path.dirname(__file__)

DEBUG = True

#Period for the system run Decision Maker Module and Partitioning the DNN
DECISION_PERIOD = 30

# Edge Configuration 
HOST_EDGE = "127.0.0.1"
PORT_EDGE = 50001
URL_EDGE = "http://%s:%s"%(HOST_EDGE, PORT_EDGE)
SAVE_IMAGES_PATH_EDGE = os.path.join(DIR_NAME, "appEdge", "images")
SAVE_COMMUNICATION_TIME_PATH = os.path.join(DIR_NAME, "appEdge", "api", "communication_time", "communication_time_b_alexnet.csv")
PROCESSING_TIME_EDGE_PATH = os.path.join(DIR_NAME, "appEdge", "api", "processing_time", "processing_time_b_alexnet_cpu_edge_raspberry.csv")
PROCESSING_TIME_CLOUD_GPU_PATH = os.path.join(DIR_NAME, "appEdge", "api", "processing_time", "processing_time_b_alexnet_gpu_cloud.csv")
PROCESSING_TIME_CLOUD_CPU_PATH = os.path.join(DIR_NAME, "appEdge", "api", "processing_time", "processing_time_b_alexnet_cpu_cloud.csv")
OUTPUT_FEATURE_BYTES_SIZE = os.path.join(DIR_NAME, "appEdge", "api", "output_bytes_size", "output_feature_bytes_size_alexnet.csv")
NR_EARLY_EXITS_PATH = os.path.join(DIR_NAME, "appEdge", "api", "branchynet_data", "nr_early_exits_b_alexnet.csv")


MONITOR_BANDWIDTH_PERIOD = 60

BRANCHES_POSITIONS = [2, 5, 7]


#Cloud Configuration 
HOST_CLOUD = "127.0.0.1"
PORT_CLOUD = 4000
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)
SAVE_IMAGES_PATH_CLOUD = os.path.join(DIR_NAME, "appCloud", "images")

#Model Paths
MODEL_EDGE_PATH = os.path.join(DIR_NAME, "appEdge", "api", "models", "model_alexnet_1.pt")
MODEL_CLOUD_PATH = os.path.join(DIR_NAME, "appCloud", "api", "models", "model_alexnet_1.pt")