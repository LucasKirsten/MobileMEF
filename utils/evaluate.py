import os
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

from funcs import *

from piq import vif_p, fsim, srsim, vsi, mdsi, LPIPS, PieAPP

class Metric():
    def __init__(self, name, callable):
        self.name = name
        self.callable = callable
        self.values = []
    
    def __call__(self, *args, **kwargs):
        val = self.callable(*args, **kwargs)
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu()
        else:
            val = np.mean(val)
        self.values.append(val)
        
    @property
    def result(self):
        return np.mean(self.values)

if __name__=='__main__':
    path_gt = './SICE/Label' # path to GT images
    path_preds = r'C:\Users\lucask\Documents\MEF_models\Fast_YUV' # path to predictions
    pattern_path_inputs = r'C:\Users\lucask\Documents\MEF_models\Ours\data\{folder}\{name}.jpg' # pattern for the input images path
    
    paths_preds = glob(os.path.join(path_preds, '*.jpg'))
    
    metrics = [
        Metric('tf_ssim', tf_ssim),
        Metric('tf_ms_ssim', tf_msssim),
        Metric('tf_psnr', tf_psnr),
        Metric('tf_deltaE 2000', tf_deltaE_ciede2000),
        Metric('tf_mef_ssim', tf_mef_ssim()),
        Metric('tf_qc', tf_Qc),
        Metric('vif_p', vif_p),
        Metric('fsim', fsim),
        Metric('srsim', srsim),
        Metric('vsi', vsi),
        Metric('mdsi', mdsi),
    ]
    
    np2torch = lambda x:torch.from_numpy(np.transpose(x, [0,3,1,2])/255.)
    
    pbar = tqdm(paths_preds)
    for pred_path in pbar:
        img_pred = cv2.imread(pred_path)
        
        name = os.path.split(pred_path)[-1]
        
        img_gt = cv2.imread(os.path.join(path_gt, name.replace('.jpg', '.JPG')))[np.newaxis,...].astype('float32')
        _,h,w,c = img_gt.shape
        
        img_pred = cv2.resize(img_pred, (w,h))[np.newaxis,...].astype('float32')
        
        input1 = pattern_path_inputs.format(name='1', folder=name.split('.')[0])
        input1 = cv2.imread(input1)
        input1 = cv2.resize(input1, (w,h))[np.newaxis,...].astype('float32')
        
        input2 = pattern_path_inputs.format(name='2', folder=name.split('.')[0])
        input2 = cv2.imread(input2)
        input2 = cv2.resize(input2, (w,h))[np.newaxis,...].astype('float32')
        
        
        for metric in metrics:
            if metric.name in ['mef_ssim', 'qc']:
                img_gt = [input1, input2]
            
            elif 'tf' in metric.name:
                metric(img_gt, img_pred)
            
            else:
                metric(np2torch(img_pred), np2torch(img_gt))
            
            pbar.set_description(str({m.name:m.result for m in metrics}))
        
    print({m.name:m.result for m in metrics})
    