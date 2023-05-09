#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#Step 1 Data 

train_df = pd.read_csv(r'D:\Spring 23\EGR 598\project\Data6.csv')
train_df['label'] = train_df['label'].astype(str)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=r'D:\Spring 23\EGR 598\project\train_images\train_data6',
        x_col='filename',
        y_col='label',
        target_size=(300, 300),
        batch_size=32,
        class_mode='sparse',
        shuffle=True)


# Step 2: Build and compile the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),   # till here better accuracy      
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
           ])

history =model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
acc_list = []

# Step 3: Train the model
history =model.fit(train_generator, epochs=5)
for acc in history.history['accuracy']:
    acc_list.append(acc)  
weights = model.get_weights()

# Step 4: saving model
model.save_weights('my_model_weights.h5')
model.save('mode_for_inference.h5')

