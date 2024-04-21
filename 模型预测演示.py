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

'''加载模型只跑预测'''
seq = load_model(r'F:\convlstm训练\convlstm模型保存\model_yanshi.h5')
seq.summary()

def select_predict(is_hour = 0):
    if is_hour == 0:
        return True
    else:
        return False

#加载预测数据函数
def load_predict_data(FolderPath = None, inputDim = 24, outputDim = 1, timelist = None, dimlist = None ,is_hour = True, t = 480, hour = 24):
    if is_hour: #逐小时预测
        # 读取inpute_data文件
        block = inputDim + outputDim
        img_data = np.zeros((1, inputDim, 27, 27, len(dimlist)))
        lable_data = np.zeros((1,outputDim,27, 27))
        print(img_data.shape)
        for k in range(inputDim):
            for j in range(len(dimlist)):
                filepath = FolderPath + '\\' + dimlist[j]+'\\'+timelist[t-inputDim+k]
                TIFF_data = Image.open(filepath) 
                img_array = np.array(TIFF_data) 
                # 数据初始化
                img_array =(img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) # 图像归一化
                img_data[0,k,:,:,j] = img_array
                # 读取lable_data文件
                if dimlist[j] == 'PM25':
                    lable_index = j
        filepath = FolderPath + '\\' + dimlist[lable_index] + '\\' + timelist[t]
        TIFF_data = Image.open(filepath)
        lable_array = np.array(TIFF_data)
        lable_array = (lable_array - np.min(lable_array)) / (np.max(lable_array) - np.min(lable_array))
        lable_min = np.min(lable_array)
        lable_max = np.max(lable_array)
        lable_data[0, :, :] = lable_array

    else: #不是逐小时预测
        # 读取inpute_data文件
        block = inputDim + outputDim
        t = t-hour
        lable_min = []
        lable_max = []
        img_data = np.zeros((hour, inputDim, 27,27, len(dimlist)))
        lable_data = np.zeros((hour, 27, 27))
        print(img_data.shape)
        for i in range(hour):
            for k in range(inputDim):
                for j in range(len(dimlist)):
                    filepath = FolderPath + '\\' + dimlist[j] + '\\' + timelist[i+t-inputDim+k]
                    TIFF_data = Image.open(filepath)  # 读取图像
                    img_array = np.array(TIFF_data)  # tiff转NP
                    # 数据初始化
                    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))  # 图像归一化
                    img_data[i, k, :, :, j] = img_array
                    # 读取lable_data文件
                    if dimlist[j] == 'PM25':
                        lable_index = j
            filepath = FolderPath + '\\' + dimlist[lable_index] + '\\'  + timelist[i+t]
            TIFF_data = Image.open(filepath)
            lable_array = np.array(TIFF_data)
            lable_min.append(np.min(lable_array))
            lable_max.append(np.max(lable_array))
            # print(np.min(lable_array), np.max(lable_array))
            lable_array = (lable_array - np.min(lable_array)) / (np.max(lable_array) - np.min(lable_array))
            lable_data[i, :, :] = lable_array
    print(img_data.shape,img_data.dtype)
    print(lable_data.shape,lable_data.dtype)
    return img_data, lable_data, lable_min, lable_max
def select_predict(is_hour = 0):
    if is_hour == 0:
        return True
    else:
        return False
#一天预测：
def oneday_predict(FolderPath=None, inputDim = 24, outputDim=1, timelist=None,dimlist=None, selcet=False,t=400, hour = 24):
    oneDayTrainData, oneDayLabelData, oneDayLabel_min, oneDayLabel_max = load_predict_data(FolderPath, inputDim, outputDim, timelist,
                                                                                           dimlist, selcet,
                                                                                           t, hour)
    if selcet == True:
        new_true = np.zeros((hour, 27, 27))
        for i in range(hour):
            new_pos = seq.predict(oneDayTrainData)
            print(f'new_pos.shape:{new_pos.shape}')
            new_pos = new_pos[0, :, :, 0]
            # print(f'new_pos.shape:{new_pos.shape}')
            new_pos = np.array(new_pos)*(oneDayLabel_max-oneDayLabel_min)+oneDayLabel_min
            new_true[i, :, :] = np.array(new_pos)
    else:
        new_true = np.zeros((hour, 27, 27))
        lable_true = np.zeros((hour, 27, 27))
        new_pos = seq.predict(oneDayTrainData)
        # print(f'new_pos.shape:{new_pos.shape}')
        for i in range(hour):
            # print(i)
            new_pos1 = new_pos[i, :, :, 0]
            labledata = oneDayLabelData[i, :, :]
            new_pos1 = np.array(new_pos1) * (oneDayLabel_max[i] - oneDayLabel_min[i]) + oneDayLabel_min[i]
            labledata = np.array(labledata) * (oneDayLabel_max[i] - oneDayLabel_min[i]) + oneDayLabel_min[i]
            new_true[i, :, :] = new_pos1
            lable_true[i, :, :] = labledata
    return new_true, lable_true

# ———————————————————————————————————————————————————————————————————————————————划分数据集
inputDim = 200  # 公共周期(小时)
outputDim = 1  # 输出周期
dimlist = ["PM25","PM10","NO2","CO","SO2","TEMP","PRES","WS","PWV"] #输入要素
FolderPath = r'F:\convlstm训练\TIF文件(一个月)' #存放路径
timelist = os.listdir(FolderPath + '\\' + dimlist[0])

#——————————————————————————————————————————————————————————————————————————————预测
t = 480 #预测433h及以后step_time
step_time = 24 #预测几个小时
#选择预测方式
    #   0 逐小时预测，并使用上一个预测结果参与下一个预测
    #   1 直接预测24小时
selcet = select_predict(1)
predict_lable, True_lable = oneday_predict(FolderPath,inputDim,outputDim,timelist,dimlist,selcet, t, step_time)

#——————————————————————绘图
for j in range(step_time):
    Predic_plot = predict_lable[j, :, :]
    True_plot = True_lable[j, :, :]
    # 可视化绘制
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=200)

    # 将颜色映射到 vmin~vmax 之间
    norm = colors.Normalize(vmin=0, vmax=100)
    # 预测结果绘图
    im1 = ax1.imshow(Predic_plot, norm=norm)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Predicted')
    #plt.show()
    # 真实值结果绘图
    im2 = ax2.imshow(True_plot, norm=norm)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('True')
    fig.suptitle(j)
    plt.savefig(f'F:\\convlstm训练\\convlstm模型保存\\{j}.png', dpi=200)
plt.show()


'''
