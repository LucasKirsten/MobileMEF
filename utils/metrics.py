import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import cv2

def metricsQc(img1, img2, fused):
    
    # Get the size of img
    m, n, b = fused.shape
    m1, n1, b1 = img1.shape
    m2, n2, b2 = img2.shape
    
    if (b1 == 1) and (b2 == 1) and (b == 3):
        fused = fused[:, :, 0]
    
    m, n, b = fused.shape
    
    if b == 1:
        g = Qc(img1, img2, fused)
        res = g
    elif b1 == 1:
        g = [Qc(img1[:, :, k], img2, fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [Qc(img1[:, :, k], img2[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    
    return res

def Qc(im1, im2, fused):
    im1 = preprocess_image(im1)
    im2 = preprocess_image(im2)
    fused = preprocess_image(fused)

    mssim2, ssim_map2, sigma_XF = ssim_yang(im1, fused)
    mssim3, ssim_map3, sigma_YF = ssim_yang(im2, fused)

    simXYF = sigma_XF / (sigma_XF + sigma_YF)
    sim = simXYF * ssim_map2 + (1 - simXYF) * ssim_map3

    sim = tf.clip_by_value(sim, 0., 1.)
    sim = sim[~tf.math.is_nan(sim)]

    output = tf.reduce_mean(sim)
    return output

def preprocess_image(img):
    if img.shape[-1] == 3:
        img = rgb2gray(img)
    return tf.cast(img, 'float32')

def ssim_yang(img1, img2):
    K = [0.01, 0.03]
    L = 255

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    window = gaussian_filter(np.ones((7, 7)), 1.5)
    window /= np.sum(window)

    img1 = tf.cast(img1, 'float32')
    img2 = tf.cast(img2, 'float32')

    mu1 = tf.nn.conv2d(img1[tf.newaxis,...], window[...,tf.newaxis,tf.newaxis], strides=[1, 1, 1, 1], padding='VALID')[0, :, :, 0]
    mu2 = tf.nn.conv2d(img2[tf.newaxis,...], window[...,tf.newaxis,tf.newaxis], strides=[1, 1, 1, 1], padding='VALID')[0, :, :, 0]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1[tf.newaxis,...]**2, window[..., tf.newaxis,tf.newaxis], strides=[1, 1, 1, 1], padding='VALID')[0, :, :, 0] - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2[tf.newaxis,...]**2, window[..., tf.newaxis,tf.newaxis], strides=[1, 1, 1, 1], padding='VALID')[0, :, :, 0] - mu2_sq
    sigma12 = tf.nn.conv2d((img1 * img2)[tf.newaxis, :, :], window[...,tf.newaxis,tf.newaxis], strides=[1, 1, 1, 1], padding='VALID')[0, :, :, 0] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = tf.reduce_mean(ssim_map)

    return mssim, ssim_map, sigma12

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / (gauss.sum())


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, window_size/6.).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(1, channel, window_size, window_size).contiguous()) / channel
    return window

def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2).view(K, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2).view(K, H, W) \
                        - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2).view(H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2).view(H, W) - muX_sq

    # compute correlation term
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2).view(K, H, W) \
                - muX.expand_as(muY_seq) * muY_seq

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q

def mef_ssim(X, Ys, window_size=11, is_lum=False):
    (_, channel, _, _) = Ys.size()
    window = create_window(window_size, channel)

    if Ys.is_cuda:
        window = window.cuda(Ys.get_device())
    window = window.type_as(Ys)

    return _mef_ssim(X, Ys, window, window_size, 0.08, 0.08, 0.01**2, 0.03**2, is_lum)

class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(X, Ys, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)

class RGBToYCbCr(object):
    def __call__(self, img):
        return torch.stack((0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
                           128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
                           128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
                          dim=1)