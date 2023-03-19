# SEASONet
The main code of the Network "SEASONet"  
The environment should be as follows:

torch 1.5.0  
torchvision 0.6.0  
pytorch-lightning 0.7.6


You can change the network settings in SEASONet/configs and train the network in SEASONet/Mask-main/train.py  
## configs
### model
arch(network name);box_nms_thresh; box_nms_thresh
### data
path(dataset file path); area_thd(pixel numbers threshold to ignore the too small buildings); seasons_mode(how the seasonal data would be used); channels(input data channel number); if_buffer(use or not use the building buffer box); buffer_storeylevel(how many pixels would the buffer box expand to the north, e.g., buffer_storeylevel/3 +1)
### training
lr; batch_size; epochs; gt_rpn_training(default as True to use the footprint data)
### test
resume(the checkpoint file to test the model)
### savepath
the path where to save the results

## Data
All the image processing can be found here in two classes: LabelTarget class and ImageProcessor class

## dataloaders
The dataloader of the network is defined here

## Mask-main
Training code and testing code

## models
The network structure is defined here. The main structure is based on Mask-RCNN
