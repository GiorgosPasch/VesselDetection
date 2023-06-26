import torch
from tqdm import tqdm
from fastai.vision.all import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

path_train = 'data/train'
path_valid = 'data/valid'

train_images = get_image_files(path_train, "/images")
valid_images = get_image_files(path_valid, "/images")


# Train Model
def train_model(model, optimizer, criterion, train_dl, model_path, epochs=10, save=True):
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch + 1, epochs))
        running_loss = 0.0
        for images, labels,_ in tqdm(train_dl):
            images = images.to('mps')
            labels = labels.to('mps')
            outputs = model(images)
            loss = criterion(outputs, torch.reshape(labels.float(), (labels.size(0), 1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    if (save):
        torch.save(model, model_path)
    return model


def evaluate(model, valid_dl):
    model.eval()
    with torch.no_grad():
        Y_actual, Y_preds = [], []
        for image, label, _ in tqdm(valid_dl):
            image = image.to('mps')
            label = label.to('mps')
            outputs = model(image)
            Y_preds.append(outputs)
            Y_actual.append(label)
        Y_actual = torch.cat(Y_actual)
        Y_preds = torch.cat(Y_preds)
    Y_actual, Y_preds = Y_actual.detach().cpu().nupmy(), Y_preds.detach().cpu().numpy()
    Y_actual, Y_preds = Y_actual.tolist(), Y_preds.tolist()




# Inference
def inference(path, model, transform):
    img = cv2.imread(str(path)).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    model = model.eval()
    bbs = []
    i = 0
    for rect in tqdm(rects):
        i+=1
        x, y, w, h = rect
        cropped = img[y:y + h, x:x + w]
        cropped = cv2.resize(cropped,[128,128])
        image = transform(cropped)
        image = image.to('mps')
        with torch.no_grad():
            label = model(image.unsqueeze(0))
            label = label.detach().cpu().numpy()
            label = label.tolist()
        label = label[0]
        if label[0] >= 0.5:
            bbs.append([rect,label[0]])
    bbs = clean_bbs(bbs)
    show_img(path, bbs)


# Calculate IOU of two bb
def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if float(boxAArea + boxBArea - interArea) == 0:
        return -20
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# UTILITIES
def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
    files in os.walk(root) for f in files if f.endswith(file_type)]


def get_data(df, img_path):
    images_bbs = {}
    for i, v in enumerate(df.values):
        image = str(img_path) + "/" + v[1] + '.jpg'
        bb = np.array([v[3], v[4], v[5], v[6]], dtype=np.float32)
        if image in images_bbs:
            images_bbs.get(image).append(bb)
        else:
            images_bbs[image] = [bb]
    return images_bbs


def get_data_class(df):
    images = []
    classes = []
    for i, v in enumerate(df.values):
        images.append(v[1])
        classes.append(v[2])
    return images, classes


def show_img(img_path, bbs):
    im = Image.open(img_path)
    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(im)

    for bb in bbs:
        x, y, w, h = bb[0]
        # Create a Rectangle patch
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


def clean_bbs(bbs):
    print('original bbs {}'.format(len(bbs)))
    bbs = sort(bbs)
    new_bbs = []
    for bb in bbs:
        new_bb = True
        for n_bb in new_bbs:
            iou = bb_iou(bb[0], n_bb[0])
            if iou > 0.5:
                new_bb = False
        if new_bb:
            new_bbs.append(bb)

    print('cleaned bbs {}'.format(len(new_bbs)))
    return new_bbs



def sort(sub_li):
    sub_li.sort(key=lambda x: x[1], reverse=True)
    return sub_li
