from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, add, Reshape
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16


IMAGE_SIZE=224
input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)

def Net():
    
    model=VGG16(
        weights='imagenet',include_top=False, input_shape=input_shape)
    
    for layer in model.layers:
        layer.trainable=False
        
    flat1=Flatten()(model.layers[-1].output)
    den1=Dense(256,activation='relu')(flat1)
    output=Dense(1,activation='sigmoid')(den1)
    
    model=Model(inputs=model.inputs, outputs=output)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model



