from __future__ import print_function

import os
import sys
import glob
from keras.layers import Dense
import tensorflow as tf
from keras.layers import GlobalMaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.models import Model
import pydot_ng as pydot
#from keras.applications import resnet50
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from model import resnet50
sys.path.append("../")
sys.path.append("D:/Graphviz2.38/bin")
from model.Config import configRes

print(pydot.find_graphviz())



def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

nb_train_samples = get_nb_files(configRes.train_dir)      # 训练样本个数
nb_classes = len(glob.glob(configRes.train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(configRes.val_dir)       #验证集样本个数
nb_epoch = int(configRes.nb_epoch)                # epoch数量
batch_size = int(configRes.batch_size)

#　图片生成器
train_datagen =  ImageDataGenerator(
preprocessing_function=preprocess_input,
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)
test_datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
configRes.train_dir,
target_size=(configRes.IM_WIDTH, configRes.IM_HEIGHT),
batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
configRes.val_dir,
target_size=(configRes.IM_WIDTH, configRes.IM_HEIGHT),
batch_size=batch_size)








def add_layers(model,nb_classes):
    x = model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(configRes.FC_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    m = Model(input = model.input,output = predictions)
    return m



def setup_to_tranfer_learn(model,base_model):
    for l in base_model.layers:
        l.trainable = False
    model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

def setup_to_finetune(model,freeze_num):
    for l in model.layers[:freeze_num]:
        l.trainable = False
    for l in model.layers[freeze_num:]:
        l.trainable = True
    model.compile(optimizer=optimizers.SGD(lr=configRes.lr, momentum=configRes.momentum),
    loss='categorical_crossentropy',
    metrics=['accuracy'])





model = resnet50.ResNet50(include_top = False,weights='imagenet')
'''plot_model(model,
               to_file='model.png',
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB')'''
new_model = add_layers(model,nb_classes)
#new_model.load_weights('../n.h5')

setup_to_tranfer_learn(new_model,model)

history_tl = new_model.fit_generator(
train_generator,
nb_epoch=nb_epoch,
samples_per_epoch=nb_train_samples,
validation_data=validation_generator,
nb_val_samples=nb_val_samples,
class_weight='auto')



setup_to_finetune(new_model,configRes.NB_IV3_LAYERS_TO_FREEZE)

history_ft = new_model.fit_generator(
train_generator,
nb_epoch=nb_epoch,
samples_per_epoch=nb_train_samples,
validation_data=validation_generator,
nb_val_samples=nb_val_samples,
class_weight='auto')




def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    #plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    #plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


new_model.save(configRes.resnet_model_save_dir)

if configRes.draw:
    plot_training(history_ft)