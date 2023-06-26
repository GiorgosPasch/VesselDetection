import pandas as pd
import util
from tqdm import tqdm
import cv2

images_root_path = 'data/images'
labels_root_path = 'data/labels'
target_path = 'data/targets_unbalanced.csv'
col = ["name", "class", "center_x", "center_y", "width", "height"]
num = 1


# create original targets for images
def create_target_df(cols, img_path, lbl_path):
    df = pd.DataFrame(columns=cols)
    images_path = util.filelist(img_path,'.jpg')
    for image in tqdm(images_path):
        img_dict = {}
        name = image.split('/')[2][:-4]
        lbl_file = str(lbl_path) + "/" + str(name) + ".txt"
        f = open(lbl_file, 'r')
        targets = f.read()
        targets = targets.split('\n')
        for target in targets:
            target = target.split(' ')
            if len(target) == 5:
                img_dict['name'] = name
                img_dict['class'] = target[0]
                img_dict['center_x'] = float(target[1])
                img_dict['center_y'] = float(target[2])
                img_dict['width'] = float(target[3])
                img_dict['height'] = float(target[4])
                df.loc[len(df)] = img_dict
    return df


def selectivesearch(image_path, gt_bbs):
    global num
    image = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    real_h,real_w,_ = image.shape
    negative_i = 0
    for i, rect in (enumerate(rects)):
        x, y, w, h = rect
        if w < 70 or h < 70:
            continue
        max_iou = 0
        for gt_bb in gt_bbs:
            x_center,y_center,width,height = gt_bb
            bb = [(x_center - width/2)*real_w , (y_center - height/2)*real_h, width*real_w, height*real_h]
            iou = util.bb_iou(rect, bb)
            if iou == -20:
                continue
            if iou > max_iou:
                max_iou = iou
        cropped = image[y:y+h, x:x+w]

        if(max_iou>=0.5):
            cv2.imwrite('data/images_positive/ship_'+str(num)+'.jpg', cropped)
            num = num +1
        else:
            if(negative_i<5):
                negative_i = negative_i+1
                cv2.imwrite('data/images_negative/ship_'+str(num)+'.jpg', cropped)
                num = num + 1


def search():
    df = pd.read_csv('data/original/targets.csv')
    img_bbs = util.get_data(df,'data/original/images')
    for k in tqdm(img_bbs):
        selectivesearch(k, img_bbs.get(k))


def create_final_dataset():
    df = pd.DataFrame(columns=['path', 'class'])
    positives = util.filelist('data/images_positive', '.jpg')
    negatives = util.filelist('data/images_negative', '.jpg')
    for f in positives:
        df.loc[len(df)] = {'path':f, 'class':1}
    i=0
    limit = len(2*positives)
    for f in negatives:
        img = cv2.imread(f)
        h,w,_ = img.shape
        df.loc[len(df)] = {'path': f, 'class': 0}
        i+=1
        if i == limit:
            break
    df.to_csv(target_path)


def resize():
    df = pd.read_csv(target_path)
    images, classes = util.get_data_class(df)
    for i in range(len(images)):
        img = cv2.imread(images[i])
        h,w,_ = img.shape
        resize_img = cv2.resize(img,[128,128])
        cv2.imwrite('data/img_resized_unbalanced/'+images[i].split('/')[2], resize_img)
