import cv2
import numpy as np
import tensorflow as tf

def tf_ssim(x,y):
    return tf.reduce_mean(tf.image.ssim(x,y,255.))

def tf_msssim(x,y):
    return tf.reduce_mean(tf.image.ssim_multiscale(x,y,255.))
    
def get_lpips():
    import torch
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')
    
    def __metric(y_true, y_pred):
    
        tf2torch = lambda x:torch.from_numpy(np.transpose(x, [0,3,1,2]))
        
        y_true = y_true[...,::-1]
        y_true = tf2torch(y_true/127.5 - 1.)
        
        y_pred = y_pred[...,::-1]
        y_pred = tf2torch(y_pred/127.5 - 1.)
    
        return loss_fn_alex(y_true, y_pred).cpu().detach().numpy()
        
    return __metric

def tf_mef_ssim():
    import torch
    from metrics import MEFSSIM, RGBToYCbCr
    _mef_ssim = MEFSSIM(is_lum=False)
    
    def __metric(y_true, y_pred):
        y_low, y_med = y_true
        y_low = y_low[...,::-1]
        y_med = y_med[...,::-1]
        y_true = tf.concat([y_low, y_med], axis=0)
        
        y_pred = y_pred[...,::-1]

        y_true = tf.clip_by_value(y_true/255., 0., 1.)
        y_pred = tf.clip_by_value(y_pred/255., 0., 1.)
        
        tf2torch = lambda x:torch.from_numpy(np.transpose(x, [0,3,1,2]))
        y_true = tf2torch(y_true)
        y_pred = tf2torch(y_pred)
        
        return _mef_ssim.forward(y_pred, y_true).cpu().detach().numpy()

    return __metric

def tf_Qc(y_true, y_pred):
    from metrics import metricsQc
    img1, img2 = y_true

    assert img1.shape[0]==1 and img2.shape[0]==1 and y_pred.shape[0]==1
    
    img1 = cv2.cvtColor(img1[0], cv2.COLOR_BGR2GRAY)[...,np.newaxis]
    img2 = cv2.cvtColor(img2[0], cv2.COLOR_BGR2GRAY)[...,np.newaxis]
    y_pred = cv2.cvtColor(y_pred[0], cv2.COLOR_BGR2GRAY)[...,np.newaxis]
    
    return metricsQc(img1, img2, y_pred)

def tf_psnr(x,y):
    return tf.reduce_mean(tf.image.psnr(x,y,255.))

def tf_rgb2lab(rgb):

    rgb = tf.clip_by_value(rgb, 0., 1.)
    rgb_linear = tf.pow((rgb+0.055)/1.055, 2.4)
    
    r,g,b = rgb_linear[...,0], rgb_linear[...,1], rgb_linear[...,2]
    
    X = 0.4124*r + 0.3576*g + 0.1805*b
    Y = 0.2126*r + 0.7152*g + 0.0722*b
    Z = 0.0193*r + 0.1192*g + 0.9505*b

    def __func(val):
        return tf.pow(val+tf.keras.backend.epsilon(), 1/3)
    
    L = 116*__func(Y) - 16
    A = 500*(__func(X/0.950489) - __func(Y))
    B = 200*(__func(Y) - __func(Z/1.08884))

    LAB = tf.stack([L,A,B], axis=-1)

    return LAB

def tf_sqrt(x):
    return tf.sqrt(x+tf.keras.backend.epsilon())

def hypot(x1, x2):
    return tf_sqrt(tf.square(x1) + tf.square(x2))

def rad2deg(x):
    return x*180/np.pi

def _cart2polar_2pi(x, y):
    r, t = hypot(x, y), tf.math.atan2(y, x+tf.keras.backend.epsilon())
    t += tf.where(t < 0.0, 2 * np.pi, 0)
    return r, t

def _get_dH2(lab1, lab2):
    
    _, a1, b1 = tf.unstack(lab1, axis=-1)
    _, a2, b2 = tf.unstack(lab2, axis=-1)
    
    C1 = hypot(a1, b1)
    C2 = hypot(a2, b2)

    term = (C1 * C2) - (a1 * a2 + b1 * b2)
    out = 2 * term
    
    return out

def lab2lch(lab):
    l, a, b = tf.unstack(lab, axis=-1)
    c, h = _cart2polar_2pi(a, b)
    return tf.stack([l,c,h], axis=-1)

def tf_deltaE_ciede94(y_true, y_pred):
    
    kH=1;kC=1;kL=1;k1=0.045;k2=0.015;EPS=tf.keras.backend.epsilon()

    y_true = y_true[...,::-1]/255.
    y_true = tf_rgb2lab(y_true)
    
    y_pred = y_pred[...,::-1]/255.
    y_pred = tf_rgb2lab(y_pred)

    y_true = lab2lch(y_true)
    L1, C1, _ = tf.unstack(y_true, axis=-1)

    y_pred = lab2lch(y_pred)
    L2, C2, _ = tf.unstack(y_pred, axis=-1)

    dL = L1 - L2
    dC = C1 - C2
    dH2 = _get_dH2(y_true, y_pred)

    SL = 1
    SC = 1 + k1 * C1
    SH = 1 + k2 * C1

    dE2 = tf.square(dL / (kL * SL + EPS))
    dE2 += tf.square(dC / (kC * SC + EPS))
    dE2 += tf.square(dH2 / (kH * SH + EPS))

    dE2 = tf.clip_by_value(dE2, 0., dE2.dtype.max)
    return tf_sqrt(dE2)

def tf_deltaE_ciede2000(y_true, y_pred):
    y_pred = tf.image.resize(y_pred, (tf.shape(y_true)[1],tf.shape(y_true)[2]))
    
    kL=1; kC=1; kH=1; EPS=tf.keras.backend.epsilon()

    y_true = y_true[...,::-1]/255.
    y_true = tf_rgb2lab(y_true)
    
    y_pred = y_pred[...,::-1]/255.
    y_pred = tf_rgb2lab(y_pred)

    L1, a1, b1 = tf.unstack(y_true, axis=-1)
    L2, a2, b2 = tf.unstack(y_pred, axis=-1)

    Cbar = 0.5 * (hypot(a1, b1) + hypot(a2, b2))
    c7 = tf.pow(Cbar, 7)
    G = 0.5 * (1 - tf_sqrt(c7 / (c7 + 25**7)))
    scale = 1 + G
    C1, h1 = _cart2polar_2pi(a1 * scale, b1)
    C2, h2 = _cart2polar_2pi(a2 * scale, b2)

    Lbar = 0.5 * (L1 + L2)
    tmp = tf.square(Lbar - 50)
    SL = 1 + 0.015 * tmp / (tf_sqrt(20 + tmp) + EPS)
    L_term = (L2 - L1) / (kL * SL + EPS)

    Cbar = 0.5 * (C1 + C2)  # new coordinates
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC + EPS)

    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2

    dH = tf.identity(h_diff)
    dH -= tf.where(h_diff > np.pi, 2 * np.pi, 0.0)
    dH += tf.where(h_diff < -np.pi, 2 * np.pi, 0.0)
    dH = tf.where(CC == 0.0, 0.0, dH)  # if r == 0, dtheta == 0
    dH_term = 2 * tf_sqrt(tf.clip_by_value(CC, 0, CC.dtype.max)) * tf.sin(dH / 2)

    Hbar = tf.identity(h_sum)
    mask = tf.math.logical_and(CC != 0.0, tf.abs(h_diff) > np.pi)
    mask = tf.cast(mask, 'float32')
    Hbar += tf.where(mask*h_sum < 2*np.pi, 2*np.pi, 0.0)
    Hbar -= tf.where(mask*h_sum >= 2*np.pi, 2*np.pi, 0.0)
    Hbar = tf.where(CC == 0.0, Hbar*2, Hbar)
    Hbar *= 0.5

    T = 1 - 0.17 * tf.cos(Hbar - np.deg2rad(30)) + 0.24 * tf.cos(2 * Hbar) + 0.32 * tf.cos(3 * Hbar + np.deg2rad(6)) - 0.20 * tf.cos(4 * Hbar - np.deg2rad(63))
    SH = 1 + 0.015 * Cbar * T

    H_term = dH_term / (kH * SH + EPS)

    c7 = tf.pow(Cbar, 7)
    Rc = 2 * tf_sqrt(c7 / (c7 + 25**7))
    dtheta = np.deg2rad(30) * tf.exp(-tf.square((rad2deg(Hbar) - 275) / 25))
    R_term = -tf.sin(2 * dtheta) * Rc * C_term * H_term

    dE2 = tf.square(L_term)
    dE2 += C_term**2
    dE2 += H_term**2
    dE2 += R_term
    dE2 = tf.clip_by_value(dE2, 0, dE2.dtype.max)
    ans = tf_sqrt(dE2)

    return ans