# Early-postoperative glioblastoma segmentation

A deep-learning segmentation model to segment glioblastoma on MRI images taken within 72h of the resection surgery. The predicted segmentations can be used to calculate the Extent of Resection, a useful metric to stratify patients in clinical trials. 

### Usage
If you want to infer using our model, use the segmentation pipeline in https://hub.docker.com/repository/docker/lidluq/seg-pipeline

Note that this repo contains the code used in the docker image, which is not written to be used outside of the docker enviroment. But do use this repo to report issues!

More information on how to run the pipeline can be found in the docker readme. 



