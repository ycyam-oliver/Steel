# Steel Defect Detection

(I) Objectives of the Project
========================================

In the dataset, there are \~10k steel images of size 1600x256. In the images, the steel may or may not contain defects. There are in total four different types of defects. Our task is to draw a path around the defects if there is any (Segmentation Map) and determine what type of defect it is. 

Some example of images and more details on the exploration of the data can be found in Data_Exploration.ipynb

For the final testing on Kaggle, the submission are evaluated by the Dice coefficient which compare the predicted set of pixels and the ground truth pixels.\
(link to the Kaggle competition: https://www.kaggle.com/c/severstal-steel-defect-detection/overview/description)


(II) Strategy to tackle the problem 
========================================

The task is divided into two parts.

In the first part, we first determined if the image contains any defect (binary classification) by trying transfer learning of DenseNet and VGG respectively. In the data generator, augmentation of random height/ width shift, horizontal/ vertical flip, rotation and zooming are used to improve the generalization performance of the models. 

Then for the images with any defects, we trained a U-net model for the classification of its type and drawing of the path around it (Segmentation map).

So for the test images, we will first determine if they have defects. Then for those believed to have defects, we will use the trained Unet model to predict its defect types and do the segmentation map. 



(III) Structure of the Directories
========================================

'''
Steel
└── Data_Exploration.ipynb # contains more analysis of the images we have and train/test split for the images for models training
├── data.py # define the data generator for the models 
├── data
│   ├── train_images # for storage of images for training and validation
│   ├── test_images # for storage of images for testing
│   ├── sample_submission.csv # contains all the filenames for test images
│   └── train.csv # contains all the filenames of train images with defects including their defect types and the segmentaion map of their location
├── 0_organized_data # for storing processed and resized images from train/val/test split in data.py
│   ├── train # resized training images for the training of DenseNet and VGG model
│   ├── val # resized validation images for the training of DenseNet and VGG model
│   ├── test # resized test images for the evaluation of the DenseNet and VGG model
│   └── mask_train_val_split.npz # train/val split of the training images for the Unet m odel
├── 1a_DenseNet_Kaiming
│   ├── main.ipynb # main file for the training of the DenseNet model to determine if there are defects in the images
│   └── model.py # define the DenseNet model with the desired fully connected head for classfication
├── 1b_VGG
│   ├── main.ipynb # main file for the training of the VGG model to determine if there are defects in the images
│   └── model.py # define the VGG model with the desired fully connected head for classfication
├── 2_U-net
│   ├── main.ipynb # main file for the training of the Unet model for the classification of the defect types and segmentation map of the defects
│   └── model.py # define the Unet model
└── 3_predictions
│   ├── Predict_VGG+Unet.ipynb # first determine if an image contains defect by VGG and then make classification predictiona and do segmentation map by Unet
│   ├── Predict_DesNet+Unet.ipynb # first determine if an image contains defect by DenseNet and then make classification predictiona and do segmentation map by Unet
│   └── csvfiles # contain the predictions from the two above ipynb
''' 

** training and testing images in 'data' can be downloaded from the Kaggle page

(III) More descriptions and training results of different models
========================================

\* Testing scores in the following are evaluated by Kaggle, which is the average of dice coefficient over different <ImageId, ClassId> pair in the test set.<br><br>

PART (i). determining if an image has defect
----------------------------------------------------------------------

[1a] DenseNet_Kaiming 

Transfer learning of the DenseNet121 model with the head replaced by two fully connected layers for the binary classification (i.e. whether there is defect or not). Every layer of the model is trained. 

Both the validation score of the binary classification here and the private/public score in the second stage of Unet predictions are improved if Kaiming initialiozation is used. 

Resulting (best) accuracy on<br> 
&emsp;Validation: 0.95386
&emsp;Testing (the pseudo-test set from test/train split): 0.95


[1b] VGG

Same as the [1a] above except VGG model is used as the backbone here.

Although the validation score of the binary classification is improved if Kaiming initialization is used, the private/public score in the second stage of Unet predictions would drop a bit with that. So Kaiming iniitialization is not employed here.

Resulting (best) accuracy on<br> 
&emsp;Validation: 0.90971
&emsp;Testing (the pseudo-test set from test/train split): 0.91


PART (ii). classification of the defects and semgentation map
----------------------------------------------------------------------

The U_net model has the character of having two layers of small filters (3x3) for a larger receptive field while keeping the number of parameters relatively low. So it is chosen to perform the classification of the defect type the  while doing segmentation map of the defects.

Resulting (least) loss on<br> 
&emsp;Validation: 0.02395

(IV) Results on Kaggle
========================================

[A] Predict_DenseNet+Unet
----------------------------------------------------------------------
Private Score: 0.82175
Public Score: 0.83428

[B] Predict_VGG+Unet
----------------------------------------------------------------------
Private Score: 0.80680
Public Score: 0.80314

