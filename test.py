import util
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from VesselDataset import transform
from Model import CNN
import torch

model_path = 'model/myCNN_unbalanced_100_32_01.pt'

model = CNN()
model = torch.load(model_path)
model.to('mps')

df = pd.read_csv('data/test_targets.csv')
img_bbs = util.get_data(df, 'data/test')
success = 0
total = 0
for en, k in enumerate(img_bbs):
    print('img {}'.format(en))
    img = cv2.imread(str(k)).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    real_h, real_w, _ = img.shape
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    model = model.eval()
    bbs = []
    gt_bbs = img_bbs.get(k)
    for rect in tqdm(rects):
        x, y, w, h = rect
        cropped = img[y:y + h, x:x + w]
        cropped = cv2.resize(cropped, [128, 128])
        image = transform(cropped)
        image = image.to('mps')
        with torch.no_grad():
            label = model(image.unsqueeze(0))
            label = label.detach().cpu().numpy()
            label = label.tolist()
        label = label[0][0]
        if label >= 0.5:
            total +=1
            max_iou = 0
            for gt_bb in gt_bbs:
                x_center, y_center, width, height = gt_bb
                bb = [(x_center - width / 2) * real_w, (y_center - height / 2) * real_h, width * real_w,
                      height * real_h]
                iou = util.bb_iou(rect, bb)
                if iou == -20:
                    continue
                if iou > max_iou:
                    max_iou = iou
            if max_iou>0.5:
                success += 1
print(success/total)
