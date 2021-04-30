import numpy as np 
import pandas as pd 
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils
from math import sin, cos
from conf import *
from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
from multiprocessing import Pool
import functools

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def imread(path, fast_mode=False):
    '''
    Basically cv2.imread with BGR to RGB
    '''
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    prediction string to coordinates. The coordinates will be in a dictionary
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

# variant of the above
def str2coords_valid(s, names):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
    return coords

def rotate(x, angle):
    '''
    Function that rotate roll, for orientation
    '''
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_img_coords(s):
    '''
    [✔️] Convert prediction string to image coordinates in the image
    input:
        s: prediction string
    output:
        img_xs: x on the image
        img_ys: y on the image
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

def XYZ2UV(x,y,z):
    '''
    [✔️] World Coordinate to image Coordinates
    '''
    u = args.FX * x / z + args.CX
    v = args.FY * y / z + args.CY
    return u,v

def UVZ2XY(u,v,z):
    '''
    [✔️] Image Coordinates to World Coordinates 
    '''
    x = z * (u - args.CX) / args.FX
    y = z * (v - args.CY) / args.FY
    return x,y

def VU2hmVU(v,u): 
    '''
    [✔️] In Image Coordinates, VU on the image to VU on the heatmap
    '''
    hm_V = (v - args.ORIG_H // 2) * args.IMG_HEIGHT / (args.ORIG_H // 2) / args.MODEL_SCALE
    hm_U = (u + args.MARGIN_W) * args.IMG_WIDTH  / (args.ORIG_W + 2*args.MARGIN_W) / args.MODEL_SCALE
    return hm_V, hm_U

def hmVU2VU(hm_v_float, hm_u_float):
    '''
    [✔️] In Image Coordinates, VU on the heatmap to VU on the image
    '''
    v = args.ORIG_H // 2 + hm_v_float * args.MODEL_SCALE / args.IMG_HEIGHT * (args.ORIG_H // 2)
    u = hm_u_float * args.MODEL_SCALE * (args.ORIG_W + 2*args.MARGIN_W) / args.IMG_WIDTH -args. MARGIN_W
    return v, u

def _regr_preprocess(regr_dict, vdiff, udiff):
    '''
    [✔️] preprocessing for regression
    input:
        regr_dict: a dictionary for regr contains id, yaw, pitch, roll, x, y, z
    output:
        regr_dict: modified dictionary with vdiff, udiff, roll, pitch_sin, pitch_cos, log(z)
    '''
    regr_dict["vdiff"] = vdiff
    regr_dict["udiff"] = udiff

    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])

    # follow original center3D paper appendix B: 3D BBOX Estimation Details
    regr_dict["z"] = np.log(regr_dict["z"])
    
    regr_dict.pop('x')
    regr_dict.pop('y')
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(regr_dict, hm_V_pos, hm_U_pos):
    '''
    [✔️] Recover from heatmap
    input:
        regr_dict: a dictionary
        hm_V_pos: V position on the heatmap
        hm_U_pos: U position on the heatmap
    output:
        regr_dict: regr backed dictionary
    '''
    regr_dict["z"] = np.exp(regr_dict["z"]) # invsigmoid(1/(regr_dict["z"]+1))
    _v, _u = hmVU2VU( hm_V_pos + regr_dict["vdiff"], hm_U_pos + regr_dict["udiff"])
    regr_dict["x"], regr_dict["y"] = UVZ2XY(_u, _v, regr_dict["z"])
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    
    return regr_dict

def preprocess_image(img, flip=False):
    '''
    [✔️] preprocess images, copied from public
    input:
        img: original image
        flip: hflip for augmentation
    output:
        img with pixel values between 0 and 1
    '''
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :args.MARGIN_W]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (args.IMG_WIDTH, args.IMG_HEIGHT))
    if flip: # flip augmentation
        img = img[:,::-1]
    return (img / 255).astype('float32')

def preprocess_mask_image(img): 
    '''
    [✔️] preprocess masks (things we are not interested in)
    '''
    img = img[img.shape[0] // 2:]
    bg = np.zeros_like(img).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 4]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (args.IMG_WIDTH, args.IMG_HEIGHT))  
    return (img / 255).astype('float32')

# https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
def draw_msra_gaussian(heatmap, center, sigma):
    '''
    [✔️] draw gaussian heatmap (helper)
    '''
    tmp_size = np.ceil(sigma * 3).astype(int) # fix
    mu_x = int(center[0])
    mu_y = int(center[1])
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
      heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
      g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def gaussian2D(shape, sigma=1):
    '''
    [✔️] gaussian 2D assignment around the keypoint
    '''
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def make_heatmap(m, v_arr, u_arr, z_arr):
    '''
    [✔️] draw gaussian heatmap (main)
    '''
    for v,u,z in zip(v_arr, u_arr, z_arr):
        sigma = 800 / 3.  / z / args.MODEL_SCALE
        m = draw_msra_gaussian(m, (u,v), sigma)
    return m

def get_hm_and_regr(img, labels, flip=False):
    '''
    [✔️] get heatmap and regression maps.
    input:
        image: preprocessed images, c x h x w
        labels: prediction strings from the csv
        flip: flip augmentation
    output:
        heatmap: heatmap given by gaussian 2d
        regr: regression mask
    '''
    # intialize heatmap and regr
    hm = np.zeros([args.IMG_HEIGHT // args.MODEL_SCALE, args.IMG_WIDTH // args.MODEL_SCALE], dtype='float32')
    regr = np.zeros([args.IMG_HEIGHT // args.MODEL_SCALE, args.IMG_WIDTH // args.MODEL_SCALE, 7], dtype='float32')

    # prediction string to coordinates 
    coords = str2coords(labels) 
    # get x and y on the image, using the given camera params
    xs, ys = get_img_coords(labels)
    # get z, depth
    zs = [e["z"] for e in coords]

    # from paper https://arxiv.org/pdf/1904.07850.pdf, section 3: L_offset
    # convert image coordinates to it's low-resolution  equivalent
    hm_V_arr, hm_U_arr = VU2hmVU( ys, xs )
    hm_V_arr_floor = np.floor( hm_V_arr ).astype('int') # low-resolution  equivalent
    hm_U_arr_floor = np.floor( hm_U_arr ).astype('int') # low-resolution  equivalent
    hm_V_diff = hm_V_arr - hm_V_arr_floor # we regress on this
    hm_U_diff = hm_U_arr - hm_U_arr_floor # we regress on this
    # make heatmap
    heatmap = make_heatmap(hm, hm_V_arr_floor, hm_U_arr_floor, zs)
    # construct regr_dict
    for hm_V, hm_U, vdiff,udiff, regr_dict in zip(hm_V_arr_floor, hm_U_arr_floor,hm_V_diff,hm_U_diff, coords):
        if hm_V >= 0 and hm_V < args.IMG_HEIGHT // args.MODEL_SCALE and hm_U >= 0 and hm_U < args.IMG_WIDTH // args.MODEL_SCALE:
            regr_dict = _regr_preprocess(regr_dict, vdiff, udiff)
            regr[hm_V, hm_U] = [regr_dict[n] for n in sorted(regr_dict)]
    # hflip, we need to do that for heatmap and regr as well
    if flip: 
        heatmap = np.array(heatmap[:,::-1])
        regr = np.array(regr[:,::-1])
    return heatmap, regr 

def sigmoid(x):
    '''
    [✔️] sigmoid function for regression
    '''
    return 1 / (1 + np.exp(-x))

def invsigmoid(x):
    '''
    [✔️] inverse sigmoid to regr back
    '''
    return np.log(x / (1 -x))

def postprocess_heatmap(logits, thresh=0.45):
    '''
    [✔️] This is like an NMS
    '''
    prob = sigmoid(logits)
    mp2d = torch.nn.MaxPool2d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)
    out = mp2d( torch.Tensor([[prob]]) ).numpy()[0][0]
    return (prob == out) & (prob > thresh)

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    '''
    [✔️] convert to image coord
    '''
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0):
    # not used
    def distance_fn(xyz):
        x, y, z = xyz
        x, y = convert_3d_to_2d(x, y, z0)
        y, x = x, y
        x = (x - args.IMG_SHAPE[0] // 2) * args.IMG_HEIGHT / (args.IMG_SHAPE[0] // 2) / args.MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + args.IMG_SHAPE[1] // 4) * args.IMG_WIDTH / (args.IMG_SHAPE[1] * 1.5) / args.MODEL_SCALE
        y = np.round(y).astype('int')
        return (x-r)**2 + (y-c)**2

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z0

def clear_duplicates(coords):
    '''
    [✔️] clear duplicates for post processing, this is not nms or maxpooling
    '''
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < args.DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

REGR_TARGETS = sorted( ["vdiff", "udiff" ,"z" , "yaw","pitch_sin", "pitch_cos", "roll"] )

def extract_coords(prediction, ignore_mask):
    '''
    [✔️] Coordinates extraction
    input:
        prediction: prediction that contains logits and regression targets
        ignore_mask: a mask that contains targets that we are not interested in
    output:
        coords: predicted values, we will turn them to strings later
    '''
    assert ignore_mask.shape[0] == args.ORIG_H  
    logits = prediction[0]
    regr_output = prediction[1:]
    points_mat = postprocess_heatmap(logits, thresh = args.heatmap_threshold) 
    points = np.argwhere( points_mat > 0 )
    coords = []
    for v, u in points:           
        regr_dict = dict(zip(REGR_TARGETS, regr_output[:, v, u]))
        regr_backed = _regr_back(regr_dict, v, u)
        _U, _V = XYZ2UV(regr_backed["x"], regr_backed["y"], regr_backed["z"])
        _U, _V = int(_U), int(_V)
        # determine if it's in the ignored mask
        if _V>=0 and _V < args.ORIG_H and _U>=0 and _U < args.ORIG_W and ignore_mask[_V,_U] > 0.5:  
            continue # if it is, we don't add it to coords
        coords.append(regr_backed)
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[v, u])) # sigmoid, whether it's a car or not
        coords = clear_duplicates(coords)
    return coords

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    '''
    [✔️] turn coords prediction to string
    input:
        coords: predicted values, we will turn them to strings here
    output:
        predicted strings
    '''
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

# calculate MAP, taken from Tito kaggle's map scripts
# link: https://www.kaggle.com/its7171/metrics-evaluation-script
def expand_df(df, PredictionStringCols):
    df = df.dropna().copy()
    df['NumCars'] = [int((x.count(' ')+1)/7) for x in df['PredictionString']]

    assert len(PredictionStringCols) == 7
    idarr = []
    tmparr = []
    for imgid, predstr in zip( df['ImageId'], df['PredictionString']):
        if predstr == "":
            continue
        coords = np.array(predstr.split(' ')).reshape(-1,7).astype(float)
        for cor in coords:
            idarr.append(imgid)
            tmparr.append(  cor.tolist() )
    
    prediction_strings_expanded = np.array(tmparr)
    df = pd.DataFrame(
        {
#            'ImageId': image_id_expanded,
            'ImageId': idarr,
            PredictionStringCols[0]:prediction_strings_expanded[:,0],
            PredictionStringCols[1]:prediction_strings_expanded[:,1],
            PredictionStringCols[2]:prediction_strings_expanded[:,2],
            PredictionStringCols[3]:prediction_strings_expanded[:,3],
            PredictionStringCols[4]:prediction_strings_expanded[:,4],
            PredictionStringCols[5]:prediction_strings_expanded[:,5],
            PredictionStringCols[6]:prediction_strings_expanded[:,6]
        })
    return df

def TranslationDistance(p,g, abs_dist = False):
    dx = p['x'] - g['x']
    dy = p['y'] - g['y']
    dz = p['z'] - g['z']
    diff0 = (g['x']**2 + g['y']**2 + g['z']**2)**0.5
    diff1 = (dx**2 + dy**2 + dz**2)**0.5
    if abs_dist:
        diff = diff1
    else:
        diff = diff1/diff0
    return diff

def RotationDistance(p, g):
    true=[ g['pitch'] ,g['yaw'] ,g['roll'] ]
    pred=[ p['pitch'] ,p['yaw'] ,p['roll'] ]
    q1 = R.from_euler('xyz', true)
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)
    
    # in the official metrics code:
    # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
    #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
    # this code treat θ and θ+2π differntly.
    # So this should be fixed as follows.
    W = (acos(W)*360)/pi
    if W > 180:
        W = 360 - W
    return W

thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

def check_match(train_df, valid_df, idx):  # train_df==TrueLabel, valid_df==Prediction
    keep_gt=False
    thre_tr_dist = thres_tr_list[idx]
    thre_ro_dist = thres_ro_list[idx]
    train_dict = {imgID:str2coords_valid(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for imgID,s in zip(train_df['ImageId'],train_df['PredictionString'])}
    valid_dict = {imgID:str2coords_valid(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score']) for imgID,s in zip(valid_df['ImageId'],valid_df['PredictionString'])}
    result_flg = [] # 1 for TP, 0 for FP
    scores = []
    MAX_VAL = 10**10
    for img_id in valid_dict:
        for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
            # find nearest GT
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate(train_dict[img_id]):
                tr_dist = TranslationDistance(pcar,gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    min_ro_dist = RotationDistance(pcar,gcar)
                    min_idx = idx
                    
            # set the result
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    train_dict[img_id].pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            scores.append(pcar['carid_or_score'])
    
    return result_flg, scores

_train_df = pd.read_csv(args.PATH + 'train.csv')

def calc_map(valid_df):
    if np.all( valid_df.dropna().PredictionString == "" ):  # no pred
        return 0.0

    expanded_valid_df = expand_df(valid_df, ['pitch','yaw','roll','x','y','z','Score'])

    val_label_df = _train_df[_train_df.ImageId.isin(valid_df.ImageId.unique())]
    # data description page says, The pose information is formatted as
    # model type, yaw, pitch, roll, x, y, z
    # but it doesn't, and it should be
    # model type, pitch, yaw, roll, x, y, z
    expanded_val_label_df = expand_df(val_label_df, ['model_type','pitch','yaw','roll','x','y','z'])

    n_gt = len(expanded_val_label_df)
    ap_list = []

    eval_func = functools.partial(check_match, val_label_df, valid_df)

    for _i in range(10):
        result_flg, scores = eval_func(_i)

        n_tp = np.sum(result_flg)
        recall = n_tp/n_gt

        ### randomized score version
        ### https://www.kaggle.com/c/pku-autonomous-driving/discussion/124489
        # ap = average_precision_score(result_flg, scores)*recall
        if False:
            ap = average_precision_score(result_flg, np.random.rand(len(result_flg)))*recall
        else: # pure precision * recall
            ap = np.mean(result_flg) * recall
        ap_list.append(ap)

    return np.mean(ap_list)

def trim_below_threth(CV_df, threth):
    cc = CV_df.copy()

    tmparr = []
    for st in cc.PredictionString:
        if st == "":
            tmparr.append("")
        else:
            r = np.array([float(e) for e in st.split(" ")]).reshape(-1,7)
            r = r[ r[:,6] >= threth, :]
            tmparr.append( " ".join( [ str(e) for e in r.flatten()] ) )
    cc.PredictionString = tmparr
    return cc

from loss import neg_loss
def criterion(prediction, heatmap, regr, regr_weight=1., mask_weight=0.5):
    mask = torch.eq(heatmap, 1)

    # keypoint loss, L_k, focal loss taken from original paper
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + \
               (1 - heatmap)**4 * pred_mask**2 * torch.log(1 - pred_mask + 1e-12) # + 1e-12 to avoid gradient overflow
    mask_loss = -mask_loss.mean(0).sum()

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
  
    # Sum
    loss = mask_weight * mask_loss + regr_weight * regr_loss
    return loss ,mask_loss , regr_loss

import random
def seed_everything(seed):
    '''
    [✔️] Set seed for reproducibility
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f'[✔️] Setting all seeds to be {seed} to reproduce.')

