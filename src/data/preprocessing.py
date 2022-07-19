from nis import match
from langcodes import best_match
import torch
import torchvision
import numpy as np
from scipy.optimize import linprog
import os 
from PIL import ImageDraw, Image
import json
import random
 
scale_back = lambda r, w, h: [int(r[0]*w), int(r[1]*h), int(r[2]*w), int(r[3]*h)]
center = lambda r: ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)

def match_pred_w_gt(bbox_preds : torch.Tensor, bbox_gts : torch.Tensor):
    bbox_iou = torchvision.ops.box_iou(boxes1=bbox_preds, boxes2=bbox_gts)
    bbox_iou = bbox_iou.numpy()

    A_ub = np.zeros(shape=(bbox_iou.shape[0] + bbox_iou.shape[1], bbox_iou.shape[0] * bbox_iou.shape[1]))
    for r in range(bbox_iou.shape[0]):
        st = r * bbox_iou.shape[1]
        A_ub[r, st:st + bbox_iou.shape[1]] = 1
    for j in range(bbox_iou.shape[1]):
        r = j + bbox_iou.shape[0]
        A_ub[r, j::bbox_iou.shape[1]] = 1
    b_ub = np.ones(shape=A_ub.shape[0])

    assignaments_score = linprog(c=-bbox_iou.reshape(-1), A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="highs-ds")
    # print(assignaments_score)
    if not assignaments_score.success:
        print("Optimization FAILED")
    assignaments_score = assignaments_score.x.reshape(bbox_iou.shape)
    assignaments_ids = assignaments_score.argmax(axis=1)

    # matched
    opt_assignaments = {}
    for idx in range(assignaments_score.shape[0]):
        if (bbox_iou[idx, assignaments_ids[idx]] > 0.5) and (assignaments_score[idx, assignaments_ids[idx]] > 0.9):
            opt_assignaments[idx] = assignaments_ids[idx] 
    # unmatched predictions
    false_positive = [idx for idx in range(bbox_preds.shape[0]) if idx not in opt_assignaments]
    # unmatched gts
    false_negative = [idx for idx in range(bbox_gts.shape[0]) if idx not in opt_assignaments.values()]

    gt2pred = {v: k for k, v in opt_assignaments.items()}
    return {"pred2gt": opt_assignaments, "gt2pred": gt2pred, "false_positive": false_positive, "false_negative": false_negative}

def get_objects(path, mode):
    # TODO given a document, apply OCR or Yolo to detect either words or entities.
    return

def load_predictions(name, path_preds, path_gts, path_images):
    # TODO read txt file and pass bounding box to the other function.
    
    boxs_preds = []
    boxs_gts = []
    links_gts = []
    
    for img in os.listdir(path_images):
        w, h = Image.open(os.path.join(path_images, img)).size
        preds_name = img.split(".")[0] + '.txt'
        with open(os.path.join(path_preds, preds_name), 'r') as preds:
            lines = preds.readlines()
            boxs = list()
            for line in lines:
                scaled = scale_back([float(c) for c in line[:-1].split(" ")[1:]], w, h)
                sw, sh = scaled[2] / 2, scaled[3] / 2
                boxs.append([scaled[0] - sw, scaled[1] - sh, scaled[0] + sw, scaled[1] + sh])
            boxs_preds.append(boxs)

        gts_name = img.split(".")[0] + '.json'
        with open(os.path.join(path_gts, gts_name), 'r') as f:
            form = json.load(f)['form']
            boxs = list()
            pair_labels = []
            ids = []
            for elem in form:
                boxs.append([float(e) for e in elem['box']])
                ids.append(elem['id'])
                [pair_labels.append(pair) for pair in elem['linking']]

            for p, pair in enumerate(pair_labels):
                pair_labels[p] = [ids.index(pair[0]), ids.index(pair[1])]

            boxs_gts.append(boxs)
            links_gts.append(pair_labels)

    random.seed(42)
    rand_idx = random.randint(0, len(os.listdir(path_images)))
    img = Image.open(os.path.join(path_images, os.listdir(path_images)[rand_idx])).convert('RGB')
    draw = ImageDraw.Draw(img)

    # TODO: nodes false positive -> new class 'wrong' (all incident links are 'none')
    rand_boxs_preds = boxs_preds[rand_idx]
    rand_boxs_gts = boxs_gts[rand_idx]

    for box in rand_boxs_gts:
        draw.rectangle(box, outline='blue', width=3)
    for box in rand_boxs_preds:
        draw.rectangle(box, outline='red', width=3)
    
    d = match_pred_w_gt(torch.tensor(rand_boxs_preds), torch.tensor(rand_boxs_gts))
    print(d)
    for idx in d['pred2gt'].keys():
        draw.rectangle(rand_boxs_preds[idx], outline='green', width=3)

    link_true_pos = list()
    link_false_neg = list()
    for link in links_gts[rand_idx]:
        if link[0] in d['false_negative'] or link[1] in d['false_negative']:
            link_false_neg.append(link)
            start = rand_boxs_gts[link[0]]
            end = rand_boxs_gts[link[1]]
            draw.line((center(start), center(end)), fill='red', width=3)
        else:
            link_true_pos.append(link)
            start = rand_boxs_preds[d['gt2pred'][link[0]]]
            end = rand_boxs_preds[d['gt2pred'][link[1]]]
            draw.line((center(start), center(end)), fill='green', width=3)

    img.save('prova.png')

    return

def save_results():
    # TODO output json of matching and check with visualization of images.
    return


if __name__ == "__main__":
    # bbox_gts = torch.Tensor([[3, 3, 6, 6], [7, 7, 11, 11], [10, 10, 17, 17]])
    # bbox_preds = torch.Tensor([[1, 1, 4, 4], [5, 5, 7, 7], [15, 15, 20, 20], [2, 2, 4, 4]])

    # print(match_pred_w_gt(bbox_preds, bbox_gts))
    path_preds = '/home/gemelli/projects/doc2graph/src/data/test_bbox'
    path_images = '/home/gemelli/projects/doc2graph/DATA/FUNSD/testing_data/images'
    path_gts = '/home/gemelli/projects/doc2graph/DATA/FUNSD/testing_data/adjusted_annotations'
    load_predictions('test', path_preds, path_gts, path_images)