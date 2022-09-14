#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
import os
import sys
import soundfile


# In[2]:


import librosa
import librosa.display
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import metrics


# In[3]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[4]:


from IPython.display import Audio


# In[5]:


import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter('ignore')
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[6]:


RavdessData=r"C:\Users\Aashish\OneDrive\Documents\dsp_proj\ravdess_data_set"


# In[7]:


ravdessDirectoryList = os.listdir(RavdessData)
fileEmotion = []
filePath = []
for dir in ravdessDirectoryList:
    actor = os.listdir(RavdessData+"\\"+dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        fileEmotion.append(int(part[2]))
        filePath.append(RavdessData+"\\"+dir+"\\"+file)
emotion_df = pd.DataFrame(fileEmotion,columns = ['Emotions'])
path_df=pd.DataFrame(filePath,columns = ['Path'])
Ravdess_df = pd.concat([emotion_df,path_df],axis=1)
print(filePath)


# In[8]:


Ravdess_df.Emotions.replace({1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',6:'fear',7:'disgust',8:'surprise'},inplace=True)
Ravdess_df.head()


# In[9]:


dataPath=pd.concat([Ravdess_df],axis=0)
dataPath.to_csv(r"data_Path.csv",index=False)
dataPath.head()


# In[10]:


plt.title('Count of Emotions',size=16)
sbn.countplot(dataPath.Emotions)
plt.ylabel('Count',size=12)
plt.xlabel('Emotions',size=12)
sbn.despine(top=True,right=True,left=False,bottom=False)
plt.show()


# In[11]:


def createWaveplot(data,sr,e):
    plt.figure(figsize=(10,3))
    plt.title('Waveplot for audio with {} emotion'.format(e),size=15)
    librosa.display.waveshow(data,sr=sr)
    plt.show()


# In[12]:


def createSpectrogram(data,sr,e):
    X=librosa.stft(data)
    Xdb=librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12,3))
    plt.title('Spectrogram for audio with {} emotion'.format(e),size=15)
    librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')
    plt.colorbar()


# In[13]:


emotion='fear'
path=np.array(dataPath.Path[dataPath.Emotions==emotion])[1]
data,samplingRate=librosa.load(path)
createWaveplot(data,samplingRate,emotion)
createSpectrogram(data,samplingRate,emotion)
Audio(path)


# In[14]:


emotion='angry'
path=np.array(dataPath.Path[dataPath.Emotions==emotion])[1]
data,samplingRate=librosa.load(path)
createWaveplot(data,samplingRate,emotion)
createSpectrogram(data,samplingRate,emotion)
Audio(path)


# In[15]:


emotion='happy'
path=np.array(dataPath.Path[dataPath.Emotions==emotion])[1]
data,samplingRate=librosa.load(path)
createWaveplot(data,samplingRate,emotion)
createSpectrogram(data,samplingRate,emotion)
Audio(path)


# In[16]:


def noise(data):
    noiseAmp=0.035*np.random.uniform()*np.amax(data)
    data=data+noiseAmp*np.random.normal(size=data.shape[0])
    return data

def stretch(data,rate=0.8):
    return librosa.effects.time_stretch(data,rate)

def shift(data):
    shiftRange=int(np.random.uniform(low=-5,high=5)*1000)
    return np.roll(data,shiftRange)

def pitch(data,samplingRate,pitchFactor=0.7):
    return librosa.effects.pitch_shift(data,samplingRate,pitchFactor)


# In[17]:


path=np.array(dataPath.Path)[1]
data,sampleRate=librosa.load(path)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=data, sr=sampleRate)
Audio(path)


# In[18]:


x=noise(data)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x,sr=sampleRate)
Audio(x,rate=sampleRate)


# In[19]:


x=stretch(data)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x,sr=sampleRate)
Audio(x,rate=sampleRate)


# In[20]:


x=shift(data)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x,sr=sampleRate)
Audio(x,rate=sampleRate)


# In[21]:


x=pitch(data,sampleRate)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x,sr=sampleRate)
Audio(x,rate=sampleRate)


# In[22]:


def extractFeature(fileName,mfcc,chroma,mel):
    with soundfile.SoundFile(fileName) as soundFile:
        X=soundFile.read(dtype="float32")
        sampleRate=soundFile.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X,sr=sampleRate,n_mfcc=40).T,axis=0)
        result=np.hstack((result,mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sampleRate).T,axis=0)
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X,sr=sampleRate).T,axis=0)
        result=np.hstack((result,mel))
    return result


# In[23]:


emotions={
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}
observedEmotions=['calm','happy','fearful','disgust']


# In[24]:


def loadData(test_size=0.2):
    x,y=[],[]
    for file in glob.glob(r"C:\\Users\\Aashish\\OneDrive\\Documents\\dsp_proj\\ravdess_data_set\\Actor_*\\*.wav"):
        fileName=os.path.basename(file)
        emotion1=emotions[fileName.split("-")[2]]
        if emotion1 not in observedEmotions:
            continue
        feature=extractFeature(file,mfcc=True,chroma=True,mel=True)
        x.append(feature)
        y.append(emotion1)
    return train_test_split(np.array(x),y,test_size=test_size,random_state=9)


# In[25]:


x_train,x_test,y_train,y_test=loadData(test_size=0.20)


# In[26]:


print((x_train.shape[0],x_test.shape[0]))


# In[27]:


print(f'Features extracted: {x_train.shape[1]}')


# In[28]:


model=MLPClassifier(alpha=0.01,batch_size=256,epsilon=1e-08,hidden_layer_sizes=(300,),learning_rate='adaptive',max_iter=500)


# In[29]:


model.fit(x_train,y_train)


# In[30]:


expected_Of_y = y_test
yPred=model.predict(x_test)


# In[31]:


print(metrics.confusion_matrix(expected_Of_y,yPred))


# In[32]:


print(classification_report(y_test,yPred))


# In[33]:


accuracy=accuracy_score(y_true=y_test, y_pred=yPred)

print("Accuracy: {:.2f}%".format(accuracy*100))

