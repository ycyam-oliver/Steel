# define data generator for feeding data into model

import numpy as np
import cv2

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from glob import glob


# for (I) binary classification of the DenseNet or VGG network 
# to determine if the images has defects

df_train=pd.read_csv('../0_organized_data/train/df_train.csv',index_col=0)
df_val=pd.read_csv('../0_organized_data/val/df_val.csv',index_col=0)
df_test=pd.read_csv('../0_organized_data/test/df_test.csv',index_col=0)


# for (II) segmentation of the U-net network

train_df=pd.read_csv('../data/train.csv')
mask_count_df=train_df.groupby('ImageId')['ClassId'].count().reset_index().rename(
    columns={'ClassId':'Num_ClassId'})
#count the number of defects in descending order
mask_count_df.sort_values('Num_ClassId',ascending=False,inplace=True)


# for the submission test images without labels

submission_df=pd.read_csv('../data/sample_submission.csv')
test_df=pd.DataFrame(submission_df['ImageId'].unique(),
                     columns=['ImageId'])

IMAGE_SIZE=224
BATCH_SIZE=32

# ============================================================
# (A) We can use built-in data generators for Desnet and VGG 
# models which are just used to classify if there is defect 
# in an image (i.e. no segmentation neeeded)

# -------------------------------------
# I. Define the Image Genrators
# -------------------------------------

# Using the written generator from Keras
data_generator=ImageDataGenerator(
    zoom_range=0.1,
    fill_mode='constant',
    cval=0.,
    rotation_range=10,
    height_shift_range=0.05,
    width_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1/255);
    
val_generator=ImageDataGenerator(
    rescale=1/255);

# -------------------------------------
# II. Flow the data into the generators
# -------------------------------------

train_gen=data_generator.flow_from_dataframe(
    df_train,directory='../0_organized_data/train',
    x_col='ImageId',
    y_col='defect_label',
    class_mode='raw',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE);

val_gen=val_generator.flow_from_dataframe(df_val,
    directory='../0_organized_data/val',
    x_col='ImageId',
    y_col='defect_label',
    class_mode='raw',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE);

test_gen=val_generator.flow_from_dataframe(
    df_test,
    directory='../0_organized_data/test',
    x_col='ImageId',
    y_col='defect_label',
    class_mode='raw',
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False);

#submit test is for thee 'real test' data
submit_test_gen=ImageDataGenerator(rescale=1/255).flow_from_dataframe(
    test_df,
    directory='../data/test_images',
    x_col='ImageId',
    class_mode=None,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False);


# ============================================================
# (B) For the U-net model (for segmentation) where we cannot 
# the built-in data generator because 
# we need to make the defect detection 
# area consistent with the data rolled out

mask_load=np.load('../0_organized_data/mask_train_val_split.npz')
mask_train_idx=mask_load['mask_train_idx']
mask_val_idx=mask_load['mask_val_idx']

# some util functions for the data generator

def rle2mask(rle, input_shape):
    
    width, height=input_shape[:2]
    
    mask=np.zeros(width*height).astype(np.uint8)
    
    runs=np.asarray([int(x) for x in rle.split()])
    starts=runs[::2]
    lengths=runs[1::2]
    for i in range(len(starts)):
        mask[int(starts[i]):int(starts[i]+lengths[i])]=1
        
    return mask.reshape(height,width).T

def mask2rle(img):
    
    '''
    input:
        img is a numpy array where in it,
    1: mask
    0: background
    
    return: an array stating the starting array of the defect 
    and how long it is (i.e. rle)
        
    '''
    
    pixels=img.T.flatten()
    
    pixels=np.concatenate([[0],pixels,[0]]) 
    runs=np.where(pixels[1:]!=pixels[:-1])[0]+1
    runs[1::2]-=runs[::2]
    return ' '.join(str(x) for x in runs)

# to deal with mask with a few channels
def build_rles(masks):
    width, height, depth= masks.shape
    
    rles=[mask2rle(masks[:,:,i]) for i in range(depth)]
    
    return rles

# self defined data generator for the mask data

MASK_BATCH_SIZE=16
class Mask_DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, df, target_df=None, 
                 mode='fit',
                 base_path='../data/train_images',
                 batch_size=MASK_BATCH_SIZE, dim=(256,1600),
                 n_channels=1, n_classes=4,
                 random_state=2021, shuffle=True):
        
        self.list_IDs=list_IDs
        self.df=df
        self.target_df=target_df
        
        self.mode=mode
        self.base_path=base_path
        
        self.batch_size=batch_size #num of pics in one batch
        self.dim=dim
        
        self.n_channels=n_channels
        self.n_classes=n_classes # defaulted to have 4 classes of defects
        self.random_state=random_state
        self.shuffle=shuffle
        
        # shuffle index after the end of each epoch
        # defined below
        self.on_epoch_end()
        
    def __len__(self):
        
        'return the number of batches (in each epoch) '
        
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def __getitem__(self, index):
        
        'generate one batch of data'
        
        # generate indices in a batch
        # index refers to the index of the batch
        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # the corresponding imageIDs of the indices generated in a batch
        list_IDs_batch=[self.list_IDs[k] for k in indexes]
        
        X=self.__generate_X(list_IDs_batch) # defined below, return grayscale images of give ImageId
        
        if self.mode=='fit':
            y=self.__generate_y(list_IDs_batch) # defined below, return the mask of a given image
            return X, y
        elif self.mode=='predict':
            return X
        else:
            raise AttributeError('Unrecognized mode! It should be "fit" or "predict".')
            
    def on_epoch_end(self):
        'Update indexes after every epoch'
        
        self.indexes=np.arange(len(self.list_IDs))
        if self.shuffle==True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
            
    def __generate_X(self, list_IDs_batch):
        'return images according to the given IDs'
        X=np.empty((self.batch_size,*self.dim,self.n_channels))
        
        # give Images
        for i, ID in enumerate(list_IDs_batch):
            img_name=self.df['ImageId'].iloc[ID]
            img_path=f"{self.base_path}/{img_name}"
            img=self.__load_grayscale(img_path)
            X[i,]=img
            
        return X
    
    def __load_grayscale(self, img_path):
        
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img=img.astype(np.float32)/255.
        img=np.expand_dims(img,axis=-1) # make the extra dimension for the 'channel' dimension
        
        return img
    
    def __load_rgb(self,img_path):
        
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB)
        img=img.astype(np.float32)/255.
        
        return img
    
    def __generate_y(self,list_IDs_batch):
        'generate the coordinates of the defects (i.e. the "mask")'
        
        y=np.empty((self.batch_size,*self.dim,self.n_classes),dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            
            image_df=self.target_df[
                self.target_df['ImageId']==(self.df['ImageId'].iloc[ID])].copy().reset_index()
            
            masks=np.zeros((*self.dim,self.n_classes))
            for j in range(len(image_df)):
                rle=image_df.loc[j,'EncodedPixels']
                cls=image_df.loc[j,'ClassId']
                masks[:,:,cls-1]=rle2mask(rle,self.dim)
            
            y[i,]=masks
        
        return y


mask_train_gen=Mask_DataGenerator(mask_train_idx,
                              df=mask_count_df,
                              target_df=train_df,
                              batch_size=MASK_BATCH_SIZE)

mask_val_gen=Mask_DataGenerator(mask_val_idx,
                            df=mask_count_df,
                            target_df=train_df,
                            batch_size=MASK_BATCH_SIZE)

