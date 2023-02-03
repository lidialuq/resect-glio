import pickle

metrics_dic = '/home/lidia/CRAI-NAS/all/lidfer/Datasets/try_docker/test_metrics.pth'
#read metrics_dic
with open(metrics_dic, 'rb') as f:
    metrics = pickle.load(f)

# print keys
print(metrics.keys())
# print all values
print(metrics)
