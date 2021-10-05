from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam,Nadam

# import tensorflow.keras.experimental as optimizers

IMAGE_SIZE=224

def Net():
    
    densenet=DenseNet121(
        include_top=False,
        input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)
    )
    
    model=Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5)) # regularization 
    model.add(layers.Dense(512,activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5)) # regularization
    model.add(layers.Dense(1,activation='sigmoid'))
    
    # cos_decay=optimizers.CosineDecay(
        # initial_learning_rate=0.001,decay_steps=10)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Nadam(),
        metrics=['accuracy']
    )
    
    return model

