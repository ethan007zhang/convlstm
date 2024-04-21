import matplotlib.pyplot as plt
from matplotlib import colors
import os
import tensorflow as tf
from osgeo import gdal
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Conv2D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

'''训练模型'''
def load_img_data(FolderPath, inputDim, outputDim, dimlist):#数据路径/输入周期/输出周期/要素列表
    block = inputDim + outputDim #216+24
    # 读取train_data文件
    timelist = os.listdir(FolderPath + '\\' + dimlist[0])
    print(len(timelist)) #数据时间长度
    img_data = np.zeros((len(timelist)-block+1, inputDim, 27,27, len(dimlist))) 
    lable_data = np.zeros((len(timelist)-block+1, 27,27))
    print(img_data.shape)
    print(lable_data.shape)

    for i in range(len(timelist) - block + 1):  # 按照样本数进行填充
        for j in range(len(dimlist)):  # 循环读取要素
            for k in range(inputDim):   # 按照输入周期进行填充
                filepath = FolderPath + '\\' + dimlist[j]+'\\'+timelist[i+k]
                TIFF_data = gdal.Open(filepath)

                img_array = TIFF_data.ReadAsArray()
                diff = np.max(img_array) - np.min(img_array)
                threshold = 1e-6  # 设置一个阈值，用于检查最大值和最小值之间的差异
                if diff < threshold:
                    diff = threshold
                img_array = (img_array -np.min(img_array)) / diff

                img_data[i,k,:,:,j] = img_array
                # 读取lable_data文件
            if dimlist[j] == 'PM25':

                filepath = FolderPath + '\\' + dimlist[j] + '\\' + timelist[i+k+1]
                TIFF_data = gdal.Open(filepath)
                lable_array = TIFF_data.ReadAsArray()
                lable_array = (lable_array - np.min(lable_array)) / (np.max(lable_array) - np.min(lable_array))
                lable_data[i, :, :] = lable_array

    print(img_data.shape,img_data.dtype)
    print(lable_data.shape,lable_data.dtype)
    X_train, X_test, Y_train, Y_test = train_test_split(img_data, lable_data,
                                                        test_size=0.2,
                                                        random_state=0, shuffle=True)
    return X_train, X_test, Y_train, Y_test , timelist

# ———————————————————————————————————————————————————————————————————————————————划分数据集
inputDim = 185  # 公共周期(小时)
outputDim = 1  # 输出周期
dimlist = ["PM25","PM10","NO2","CO","SO2","PRES","WS","PWV"]
FolderPath = r'F:\convlstm训练\TIF文件(一个月)'
X_train, X_test, Y_train, Y_test ,timelist = load_img_data(FolderPath, inputDim ,outputDim, dimlist)
valdata = (X_test,Y_test)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#——————————————————————————————————————————————————————————————————————训练数据集，
# 创建convlstm模型
def create_model():
    seq = Sequential()
    #添加第一个网络层：卷积核个数和大小
    seq.add(ConvLSTM2D(filters=12, kernel_size=(3,3),
                       input_shape=(None, 27, 27, 8),
                       padding='same', return_sequences=True)) 
    #添加第二个网络层：
    seq.add(BatchNormalization()) #归一化层，加速训练

    seq.add(ConvLSTM2D(filters=18, kernel_size=(3, 3),
                       padding='same', return_sequences=False))
    seq.add(BatchNormalization())

    seq.add(Conv2D(filters=1, kernel_size=(3, 3),activation='sigmoid',padding='same',
                   data_format='channels_last')) #改卷积核大小
    seq.compile(loss='mean_squared_error', optimizer='RMSprop') #均方误差做为损失函数，优化器：RMSprop
    seq.summary()
    return seq
seq = create_model() #调用模型函数

#开始训练
result = seq.fit(X_train, Y_train, batch_size=10, epochs=50, validation_data=valdata)

#保存模型
seq.save(r'F:\convlstm训练\convlstm模型保存\model_yanshi.h5') #

# 绘制训练 & 验证的损失值
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
