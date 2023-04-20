import os
import sys
import json
import numpy as np
import math
from math import sqrt
from PIL import Image, ImageDraw
import dgl
import torch
import random

import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.getcwd())
os.environ['DGLBACKEND'] = 'pytorch'

from src.paths import *


def load_json(filename):
  with open(filename, 'r') as f:
      data = json.load(f)
  return data

def polar(rect_src : list, rect_dst : list):
    """Compute distance and angle from src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0
    bottom = (rect_src[3] - rect_dst[1]) <= 0
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 

    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

    # evaluate reciprocal position
    sc = center(rect_src)
    ec = center(rect_dst)
    new_ec = (ec[0] - sc[0], ec[1] - sc[1])
    angle = int(math.degrees(math.atan2(new_ec[1], new_ec[0])) % 360)
    
    if rect_intersect:
        return 0, angle
    elif top and left:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), angle
    elif left and bottom:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), angle
    elif bottom and right:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), angle
    elif right and top:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), angle
    elif left:
        return (rect_src[0] - rect_dst[2]), angle
    elif right:
        return (rect_dst[0] - rect_src[2]), angle
    elif bottom:
        return (rect_dst[1] - rect_src[3]), angle
    elif top:
        return (rect_src[1] - rect_dst[3]), angle


def parser_annotation_info(form):
# getting infos
  boxs, texts, ids, nl = list(), list(), list(), list()
  pair_labels = list()
  for elem in form:
      boxs.append(elem['box'])
      texts.append(elem['text'])
      nl.append(elem['label'])
      ids.append(elem['id'])
      [pair_labels.append(pair) for pair in elem['linking']]
  
  for p, pair in enumerate(pair_labels):
      pair_labels[p] = [ids.index(pair[0]), ids.index(pair[1])]   

  # get edge:
  return nl, texts, boxs, pair_labels


def knn(size : tuple, bboxs : list, k = 10):
    """ Given a list of bounding boxes, find for each of them their k nearest ones.

    Args:
        size (tuple) : width and height of the image
        bboxs (list) : list of bounding box coordinates
        k (int) : k of the knn algorithm
    
    Returns:
        u, v (lists) : lists of indices
    """

    edges = []
    width, height = size[0], size[1]
    
    # creating projections
    vertical_projections = [[] for i in range(width)]
    horizontal_projections = [[] for i in range(height)]
    for node_index, bbox in enumerate(bboxs):
        for hp in range(bbox[0], bbox[2]):
            if hp >= width: hp = width - 1
            vertical_projections[hp].append(node_index)
        for vp in range(bbox[1], bbox[3]):
            if vp >= height: vp = height - 1
            horizontal_projections[vp].append(node_index)
    
    def bound(a, ori=''):
        if a < 0 : return 0
        elif ori == 'h' and a > height: return height
        elif ori == 'w' and a > width: return width
        else: return a

    for node_index, node_bbox in enumerate(bboxs):
        neighbors = [] # collect list of neighbors
        window_multiplier = 2 # how much to look around bbox
        wider = (node_bbox[2] - node_bbox[0]) > (node_bbox[3] - node_bbox[1]) # if bbox wider than taller
        
        ### finding neighbors ###
        while(len(neighbors) < k and window_multiplier < 100): # keep enlarging the window until at least k bboxs are found or window too big
            vertical_bboxs = []
            horizontal_bboxs = []
            neighbors = []
            
            if wider:
                h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier/4)
                v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier)
            else:
                h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier)
                v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier/4)
            
            window = [bound(node_bbox[0] - h_offset),
                    bound(node_bbox[1] - v_offset),
                    bound(node_bbox[2] + h_offset, 'w'),
                    bound(node_bbox[3] + v_offset, 'h')] 
            
            [vertical_bboxs.extend(d) for d in vertical_projections[window[0]:window[2]]]
            [horizontal_bboxs.extend(d) for d in horizontal_projections[window[1]:window[3]]]
            
            for v in set(vertical_bboxs):
                for h in set(horizontal_bboxs):
                    if v == h: neighbors.append(v)
            
            window_multiplier += 1 # enlarge the window
        
        ### finding k nearest neighbors ###
        neighbors = list(set(neighbors))
        if node_index in neighbors:
            neighbors.remove(node_index)
        neighbors_distances = [polar(node_bbox, bboxs[n])[0] for n in neighbors]
        for sd_num, sd_idx in enumerate(np.argsort(neighbors_distances)):
            if sd_num < k:
                if [node_index, neighbors[sd_idx]] not in edges and [neighbors[sd_idx], node_index] not in edges:
                    edges.append([neighbors[sd_idx], node_index])
                    edges.append([node_index, neighbors[sd_idx]])
            else: break

    return [e[0] for e in edges], [e[1] for e in edges]

def fully_connected(ids : list):
  """ create fully connected graph

  Args:
      ids (list) : list of node indices
  
  Returns:
      u, v (lists) : lists of indices
  """
  u, v = list(), list()
  for id in ids:
      u.extend([id for i in range(len(ids)) if i != id])
      v.extend([i for i in range(len(ids)) if i != id])
  return u, v


def balance_edges(g : dgl.DGLGraph, cls=None ):
  """ if cls (class) is not None, but an integer instead, balance that class to be equal to the sum of the other classes

  Args:
      g (DGLGraph) : a DGL graph
      cls (int) : class number, if any
  
  Returns:
      g (DGLGraph) : the new balanced graph
  """
  
  edge_targets = g.edata['label']
  u, v = g.all_edges(form='uv')
  edges_list = list()
  for e in zip(u.tolist(), v.tolist()):
      edges_list.append([e[0], e[1]])

  if type(cls) is int:
      to_remove = (edge_targets == cls)
      indices_to_remove = to_remove.nonzero().flatten().tolist()

      for _ in range(int((edge_targets != cls).sum()/2)):
          indeces_to_save = [random.choice(indices_to_remove)]
          edge = edges_list[indeces_to_save[0]]

          for index in sorted(indeces_to_save, reverse=True):
              del indices_to_remove[indices_to_remove.index(index)]

      indices_to_remove = torch.flatten(torch.tensor(indices_to_remove, dtype=torch.int32))
      g = dgl.remove_edges(g, indices_to_remove)
      return g
  else:
      raise Exception("Select a class to balance (an integer ranging from 0 to num_edge_classes).")

def draw_graph(g,nl, boxs,  el, img_input_path, img_name, img_output_path):
    edge_unique_labels = np.unique(el)
    g.edata['label'] = torch.tensor([np.where(target == edge_unique_labels)[0][0] for target in el])
    #g = balance_edges(g, 3)

    img_removed = Image.open(img_input_path).convert('RGB')
    draw_removed = ImageDraw.Draw(img_removed)

    for b, box in enumerate(boxs):
        if nl[b] == 'header':
            color = 'yellow'
        elif nl[b] == 'question':
            color = 'blue'
        elif nl[b] == 'answer':
            color = 'green'
        else:
            color = 'gray'
        draw_removed.rectangle(box, outline=color, width=3)

    u, v = g.all_edges()
    labels = g.edata['label'].tolist()
    u, v = u.tolist(), v.tolist()

    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

    num_pair = 0
    num_none = 0

    for p, pair in enumerate(zip(u,v)):
        sc = center(boxs[pair[0]])
        ec = center(boxs[pair[1]])
        check = np.where('pair' == edge_unique_labels)
        
        if len(check[0]) == 0:  
            num_none += 1
            color='gray'

        elif labels[p] == int(check[0][0]): 
            num_pair += 1
            color = 'violet'
            draw_removed.ellipse([(sc[0]-4,sc[1]-4), (sc[0]+4,sc[1]+4)], fill = 'green', outline='black')
            draw_removed.ellipse([(ec[0]-4,ec[1]-4), (ec[0]+4,ec[1]+4)], fill = 'red', outline='black')
            draw_removed.line((sc,ec), fill=color, width=3)
        else: 
            num_none += 1
            color='gray'
        #draw_removed.line((sc,ec), fill=color, width=3)
                    
    print("Balanced Links: None {} | Key-Value {}".format(num_none, num_pair))
    img_removed.save(img_output_path)

def draw_graph_embedding(img_path,
                          boxs, srcs, dsts,
                          angles,
                          output_path):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
    for p, pair in enumerate(zip(srcs, dsts)): 
      sc = center(boxs[pair[0]])
      ec = center(boxs[pair[1]])
      draw.line((sc, ec), fill='grey', width=3)
      middle_point = ((sc[0] + ec[0])/2,(sc[1] + ec[1])/2)
      draw.text(middle_point, str(angles[p]), fill='black')
      draw.rectangle(boxs[pair[0]], fill='red')
      draw.rectangle(boxs[pair[1]], fill='blue')
    
    img.save(output_path)  


def to_bin(dist :int, angle : int, b=8) -> torch.Tensor:
    """ Discretize the space into equal "bins": return a distance and angle into a number between 0 and 1.

    Args:
        dist (int): distance in terms of pixel, given by "polar()" util function
        angle (int): angle between 0 and 360, given by "polar()" util function
        b (int): number of bins, MUST be power of 2
    
    Returns:
        torch.Tensor: new distance and angle (binary encoded)

    """
    def isPowerOfTwo(x):
        return (x and (not(x & (x - 1))) )

    # dist
    assert isPowerOfTwo(b)
    m = max(dist) / b
    new_dist = []
    for d in dist:
        bin = int(d / m)
        if bin >= b: bin = b - 1
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_dist.append(bin)
    
    # angle
    amplitude = 360 / b
    new_angle = []
    for a in angle:
        bin = (a - amplitude / 2) 
        bin = int(bin / amplitude)
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_angle.append(bin)

    return torch.cat([torch.tensor(new_dist, dtype=torch.float32), torch.tensor(new_angle, dtype=torch.float32)], dim=1)


# scaling by img width and height
sg = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] 

import spacy
text_embedder = spacy.load('en_core_web_lg')

if __name__ == '__main__':
    
    os.makedirs('mycode/images', exist_ok=True)
    os.makedirs('mycode/images_embedding', exist_ok=True)

    dataset_name = 'FUNSD'
    edge_type = 'knn' # or fully_connected
    num_polar_bins = 8 #using eweight for link
    DATA = str(DATA).replace('/code', '.')
    image_path = os.path.join(DATA,dataset_name,'training_data','images')
    annotation_path = os.path.join(DATA,dataset_name,'training_data','adjusted_annotations')

    assert os.path.exists(image_path),f'image path is not found'
    assert os.path.exists(annotation_path),f'annotaiton path is not found'

    for file in sorted(os.listdir(annotation_path)):
        img_path = os.path.join(image_path, f'{file.split(".")[0]}.png')
        ann_path = os.path.join(annotation_path, file)
        assert os.path.exists(img_path), f'{img_path} is not found'
        form = load_json(ann_path)['form']
        # nl = label in node, texts = gt, boxes = position, pair_labels = link
        nl, texts, boxs, pair_labels = parser_annotation_info(form)

        # getting edges
        if edge_type == 'knn':
          u,v = knn(Image.open(img_path).size, boxs)
        elif edge_type == 'fully_connected':
          u,v = fully_connected(range(len(boxs)))

        # el is link between node
        el = list()
        for e in zip(u, v):
            edge = [e[0], e[1]]
            if edge in pair_labels: el.append('pair')
            else: el.append('none')

        # creating graph:
        # using u,v, el boxs for creating graph    
        g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(boxs), idtype=torch.int32)
        
        img_output_path = os.path.join('mycode/images', f'{file.split(".")[0]}.png')
        draw_graph(g,nl, boxs, el, img_path, f'{file.split(".")[0]}.png' ,img_output_path)

        #img_path, texts, boxs
        # add embedding features in nodes
        #get feature:
        size = Image.open(img_path).size
        feats = [[] for _ in range(len(boxs))]
        geom = [sg(box, size) for box in boxs]
        chunks = [] 
        # 'geometrical' features
        [feats[idx].extend(sg(box, size)) for idx, box in enumerate(boxs)]
        chunks.append(4) # 4 dim
        
        # 
        # LANGUAGE MODEL (SPACY) features 
        # concatinate feature
        [feats[idx].extend(text_embedder(texts[idx]).vector) for idx, _ in enumerate(feats)]
        chunks.append(len(text_embedder(texts[0]).vector))

        # add embedding feature to edge
        # use g, boxs
        _u, _v = g.edges()
        srcs, dsts =  _u.tolist(), _v.tolist()
        distances = []
        angles = []

        for pair in zip(srcs, dsts):
          dist, angle = polar(boxs[pair[0]], boxs[pair[1]])
          distances.append(dist)
          angles.append(angle)
        m = max(distances)
        polar_coordinates = to_bin(distances, angles, num_polar_bins)

        # add features into graph
        g.edata['feat'] = polar_coordinates # number of edge x 
        g.ndata['geom'] = torch.tensor(geom, dtype=torch.float32)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)

        distances = torch.tensor([(1-d/m) for d in distances], dtype=torch.float32)
        tresh_dist = torch.where(distances > 0.9, torch.full_like(distances, 0.1), torch.zeros_like(distances))
        g.edata['weights'] = tresh_dist

        norm = []
        num_nodes = len(boxs) - 1
        for n in range(num_nodes + 1):
            neigs = torch.count_nonzero(tresh_dist[n*num_nodes:(n+1)*num_nodes]).tolist()
            try: norm.append([1. / neigs])
            except: norm.append([1.])
        g.ndata['norm'] = torch.tensor(norm, dtype=torch.float32)  

        image_output2 = os.path.join('mycode/images_embedding', f'{file.split(".")[0]}.png')
        draw_graph_embedding(img_path,
                          boxs, srcs, dsts,
                          angles,
                          image_output2)   

       

        
  


 
        