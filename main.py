import warnings
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import util
from VesselDataset import VesselDataset
from Model import CNN
from VesselDataset import transform

'''
Train model if true, evaluate if not
'''
TRAIN = False
'''
Save trained model
'''
SAVE = False

'''
Balanced between background and ship
'''
balanced = True

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.01

warnings.filterwarnings('ignore')

'''
model name : name_imgSize_epoch_batch_OtherInfo.pt
'''
if balanced:
    images_path = 'data/img_resized/'
    labels_path = 'data/targets.csv'
    model_path = 'model/myCNN_' + str(EPOCHS) + '_' + str(BATCH_SIZE) + '_' + str(LR).split('.')[1] + '.pt'
else:
    images_path = 'data/img_resized_unbalanced/'
    labels_path = 'data/targets_unbalanced.csv'
    model_path = 'model/myCNN_unbalanced_'+str(EPOCHS)+'_'+str(BATCH_SIZE)+ '_' + str(LR).split('.')[1] +'.pt'

col = ["name", "class"]

model = CNN()
model.to('mps')

if TRAIN:
    model.train()
    df_train = pd.read_csv(labels_path)
    images, bbs = util.get_data_class(df_train)
    dataset = VesselDataset(images, bbs, images_path)
    trainDataset, valDataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    criterion = nn.BCELoss()
    model = util.train_model(model, optimizer, criterion, trainLoader, model_path, epochs=EPOCHS, save=SAVE)
    util.evaluate(model, valDataset)
else:
    model = torch.load(model_path)
    model.to('mps')
    test_imgs = [
        '4ec8c94f2_jpg.rf.9ef2d31c37fc04fdc566dd3316f8a968.jpg',
        '6882479b6_jpg.rf.b63a3d7f9ece0821fc09b028f3a8d34b.jpg',
        '026bb1723_jpg.rf.fc9d43730d9736c7d58b47b47807ede5.jpg',
        '81d3bfee0_jpg.rf.cae11b570cfcca9f5159b8aec217daa7.jpg',
        '4656f24a7_jpg.rf.7e9be4aae738e7041397d8a3be692b17.jpg']
    for test_img in test_imgs:
        test_path = 'data/original/images/'+test_img
        util.inference(test_path, model, transform)