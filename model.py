import os
import cv2
import numpy as np
#import gc

#import onnx
#import tf2onnx

import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras import layers as L
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import *

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from glob import glob
from tqdm import tqdm

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        :param epsilon: Small number to avoid division by zero
        :param name: Layer name
        """
        super().__init__(**kwargs)

        self.epsilon = tf.keras.backend.epsilon()
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        params_shape = input_shape[-1:]

        # Initialize beta and gamma
        self.beta = self.add_weight('beta',
                                    shape=params_shape,
                                    initializer=tf.keras.initializers.zeros,
                                    dtype=self.dtype)
        self.gamma = self.add_weight('gamma',
                                     shape=params_shape,
                                     initializer=tf.keras.initializers.ones,
                                     dtype=self.dtype)
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs,
             **kwargs) -> tf.Tensor:
        # Calculate mean and variance
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.math.reduce_std(inputs, axis=-1, keepdims=True)
        # Normalize
        normalized = (inputs - mean) / (variance + self.epsilon)  # shape=(batch_size, channels)
        return self.gamma * normalized + self.beta  # shape=(batch_size, channels)

    def get_config(self):
        base_config = super().get_config()
        base_config['epsilon'] = self.epsilon

        return base_config

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

LayerNormalization = tf.keras.layers.BatchNormalization
gelu = tf.keras.activations.gelu

def conv_block(x, channels, *args, **kwargs):
    
    skip = x

    x = tf.keras.layers.DepthwiseConv2D((7,7), padding='same') (x)
    x = LayerNormalization() (x)
    x = tf.keras.layers.Conv2D(2*channels, (1,1), padding='same') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.gelu) (x)
    x = tf.keras.layers.Conv2D(channels, (1,1), padding='same') (x)
    
    x = tf.keras.layers.Add() ([skip, x])
    
    return x

def baseblock(x, filters, strides=(1,1)):
    out = tf.keras.layers.SeparableConv2D(filters, (4,4), padding="same", strides=strides) (x)
    
    out = conv_block(out, filters)
    out = conv_block(out, filters)

    se = tf.reduce_mean(out, axis=(1,2), keepdims=True)
    se = tf.keras.layers.Dense(x.shape[-1]*2) (se)
    se = tf.keras.layers.BatchNormalization() (se)
    se = tf.keras.layers.Activation('relu') (se)
    se = tf.keras.layers.Dense(x.shape[-1], activation='sigmoid') (se)

    out = out * se
    
    return out

def attention_block(x, dim=16):
    
    inputs = x
    shortcut = x
    
    gap = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
    gmp = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
    
    ## spatial attention
    gap_gmp = Concatenate(axis=3)([gap, gmp])
    gap_gmp = tf.keras.layers.Conv2D(dim, (3,3), strides=(1,1), 
                                     padding="same", 
                                     activation='sigmoid')(gap_gmp)
    
    spatial_attention = multiply([shortcut, gap_gmp])
    
    ## channel attention
    x1 = tf.keras.layers.Conv2D(dim, (1,1), strides=(1,1), 
                               padding="same", 
                               activation='relu')(gap)
    x1 = tf.keras.layers.Conv2D(dim, (1,1), strides=(1,1), 
                               padding="same", 
                               activation='sigmoid')(x1)
    
    channel_attention = multiply([shortcut, x1])
    
    
    attention = Concatenate(axis=3)([spatial_attention, channel_attention])
    x2 = tf.keras.layers.Conv2D(dim, (1,1), strides=(1,1), 
                               padding="same", 
                               activation=None)(attention)
    
    out = Add()([inputs, x2])
    return out

def inv_block(x, channels=3):
    m = x
    m = Conv2D(channels, (1,1), activation=None, strides=(1,1), padding='same')(m)
    m = DepthwiseConv2D((3,3), activation=None, strides=(1,1), padding='same')(m)
    m = Conv2D(channels, (1,1))(m)
    
    x = Conv2D(channels, (1,1), activation ='relu', strides=(1,1), padding='same')(x)
    y = Add()([m, x])
    return y

def sat(x, channels=3):
    f = Conv2D(channels, (7,7), padding='same', activation='relu')(x)
    f = Conv2D(channels, (5,5), padding='same', activation='relu')(f)
    f = Conv2D(channels, (3,3), activation='sigmoid', padding='same')(f)
    return x * f

def __get_model(inputs_med, inputs_low, encoder_dim, out_dim, n_encoders=3):

    encoder_dim = [encoder_dim*2**i for i in range(n_encoders)]
    shape = inputs_med.shape[1:-1]

    inputs_med = tf.image.resize(inputs_med, (shape[0]//2,shape[1]//2))
    inputs_low = tf.image.resize(inputs_low, (shape[0]//2,shape[1]//2))

    x = tf.keras.layers.Concatenate(axis=-1) ([inputs_med, inputs_low])

    encoder_fes = []

    for e in range(len(encoder_dim)):
        x = tf.keras.layers.SeparableConv2D(encoder_dim[e], (3,3), padding="same")(x)
        encoder_fes.append(x)
        
        x = tf.keras.layers.MaxPooling2D() (x)
        x = baseblock(x,encoder_dim[e])
        x = attention_block(x,encoder_dim[e])
        print ('e', e, x.shape)
    
    for d in range(len(encoder_dim)):
        cat = encoder_fes.pop()
        feats = cat.shape[-1]
        
        x = tf.keras.layers.SeparableConv2D(feats, (3,3), padding="same")(x)
        x = baseblock(x, feats)
        x = attention_block(x, feats)
        print ('d', d, x.shape)
        x = tf.keras.layers.Conv2DTranspose(feats, (3,3), strides=(2,2), padding='same')(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, cat])
    
    x = inv_block(x, out_dim)
    x = sat(x, out_dim)
    return x

def naive_multires_pyramid(image, weight_map, levels):

    def pyrUp(img):
        out = tf.image.resize(img, [img.shape[-3]*2, img.shape[-2]*2])
        return out
    
    def pyrDown(img):
        out = tf.image.resize(img, [img.shape[-3]//2, img.shape[-2]//2])
        return out
    
    levels  = levels - 1
    imgGpyr = [image]
    wGpyr   = [weight_map]
    
    for i in range(levels):
        imgGpyr.append(pyrDown(imgGpyr[i]))
        wGpyr.append(pyrDown(wGpyr[i]))

    imgLpyr = [imgGpyr[levels]]
    
    for i in range(levels, 0, -1):
        shape = imgGpyr[i-1].shape
        imgLpyr.append(imgGpyr[i-1] - tf.image.resize(pyrUp(imgGpyr[i]), shape[-3:-1]))
    
    return imgLpyr[::-1], wGpyr

def get_model(shape, batch_size=None, resize_output=False):

    HEIGHT,WIDTH = shape
    
    y_low_input  = tf.keras.layers.Input((HEIGHT, WIDTH, 1), batch_size=batch_size)
    uv_low_input = tf.keras.layers.Input((HEIGHT//4, WIDTH//4, 2), batch_size=batch_size)
    y_med_input  = tf.keras.layers.Input((HEIGHT, WIDTH, 1), batch_size=batch_size)
    uv_med_input = tf.keras.layers.Input((HEIGHT//4, WIDTH//4, 2), batch_size=batch_size)

    # SSF
    y_inputs = tf.keras.layers.Concatenate(axis=-1) ([y_low_input, y_med_input])
    y_inputs = tf.image.resize(y_inputs, (HEIGHT//2, WIDTH//2))

    u_inputs = tf.stack([uv_low_input[...,0], uv_med_input[...,0]], axis=-1)
    u_inputs = tf.image.resize(u_inputs, (HEIGHT//8, WIDTH//8))
    v_inputs = tf.stack([uv_low_input[...,1], uv_med_input[...,1]], axis=-1)
    v_inputs = tf.image.resize(v_inputs, (HEIGHT//8, WIDTH//8))
        
    Ew = tf.abs(u_inputs+tf.keras.backend.epsilon())*tf.abs(v_inputs+tf.keras.backend.epsilon())+1
    Ew = tf.image.resize(Ew, (HEIGHT//2, WIDTH//2))
    
    kernel = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]
    kernel = np.array([kernel]*2)[...,np.newaxis]
    kernel = kernel.reshape(3,3,2,1)
    Cw = tf.keras.layers.DepthwiseConv2D(
        (3,3), strides=(1,1), padding='same', depthwise_initializer=lambda shape,dtype : kernel) (y_inputs)
    Cw = tf.abs(Cw)+1

    W = Ew * Cw
    norm = tf.reduce_sum(W, axis=-1, keepdims=True)+tf.keras.backend.epsilon()
    weight_maps = W/norm
    
    Gn = tf.keras.layers.DepthwiseConv2D((5,5), strides=(1,1), padding='same', depthwise_initializer=tf.keras.initializers.Constant(value=1/25.)) (weight_maps)

    imgLpyr, wGpyr = naive_multires_pyramid(y_inputs, weight_maps, 2)
    
    L1 = imgLpyr[1]*wGpyr[1]
    L1 = tf.image.resize(L1, (HEIGHT//2, WIDTH//2))
    L1 = tf.abs(L1)

    y_inp_fuse = y_inputs *(Gn + 0.2*L1)
    y_inp_fuse = tf.reduce_sum(y_inp_fuse, axis=-1, keepdims=True)

    u_inp_fuse = tf.reduce_max(u_inputs, axis=-1, keepdims=True)
    v_inp_fuse = tf.reduce_max(v_inputs, axis=-1, keepdims=True)
    uv_inp_fuse = tf.keras.layers.Concatenate(axis=-1) ([u_inp_fuse, v_inp_fuse])
    
    # DNN
    outY = __get_model(y_med_input, y_low_input, encoder_dim=4, out_dim=1, n_encoders=5)
    outY = tf.keras.layers.Add() ([outY, y_inp_fuse])
    
    outUV = __get_model(uv_med_input, uv_low_input, encoder_dim=8, out_dim=2, n_encoders=3)
    outUV = tf.keras.layers.Add() ([outUV, uv_inp_fuse])
    
    model = tf.keras.models.Model(inputs=[y_low_input, uv_low_input, y_med_input, uv_med_input], outputs=[outY, outUV])
    return model
    
def convert_to_onnx():
    for size in [512, 768, 1024, 1280, 1536, 1792, 2048, 4096]:
        model = get_model(shape=(size,size), batch_size=1, resize_output=True)
        
        input_signature = [tf.TensorSpec(inp.shape, tf.float32, name=inp.name) for inp in model.inputs]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        onnx.save(onnx_model, f'onnx_model_{size}.onnx')
        
def convert_to_tflite():
    HEIGHT = 4096
    WIDTH  = 4096

    model = get_model(shape=(HEIGHT,WIDTH), batch_size=1, resize_output=False)
    print(model.summary())
    
    print(model.inputs)
    print(model.outputs)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(f'model.tflite', 'wb') as f:
        f.write(tflite_model)
        
def getYUVColorSpace(image):
    temp_image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(temp_image_yuv)
    return y, v, u
    
def normalizeValue(input_image):
    input_image = (input_image / 255.0)
    return input_image
        
def read_input_data(path):
    img_bgr = cv2.imread(path)
    img_bgr = cv2.resize(img_bgr, (WIDTH, HEIGHT))
    
    y, u, v = getYUVColorSpace(img_bgr)
    
    y = normalizeValue(y)
    
    u_temp_half = cv2.resize(u, (WIDTH//4, HEIGHT//4))
    v_temp_half = cv2.resize(v, (WIDTH//4, HEIGHT//4))
    
    u = normalizeValue(u_temp_half)
    v = normalizeValue(v_temp_half)
    uv = np.stack([u,v], axis=-1).astype('float32')
    
    return y[np.newaxis,...,np.newaxis], uv[np.newaxis,...,np.newaxis]
    
def Y_UV420_2_RGB(y, uv):
    y=tf.keras.backend.clip(y,0,1)
    uv=tf.keras.backend.clip(uv,0,1)
    uv_lg=tf.image.resize(uv, y.shape[1:-1])
    u, v = tf.split(uv_lg, 2, axis=3)
    target_uv_min, target_uv_max = -0.5, 0.5
    u = u * (target_uv_max - target_uv_min) + target_uv_min
    v = v * (target_uv_max - target_uv_min) + target_uv_min
    preprocessed_yuv_images = tf.concat([y, u, v], axis=-1)
    rgb_tensor= tf.image.yuv_to_rgb(preprocessed_yuv_images)
    return rgb_tensor
    
def convert_prediction(y_pred, uv_pred):
    rgb_pred = Y_UV420_2_RGB(y_pred, uv_pred)
    rgb_pred = np.squeeze(rgb_pred)
    rgb_pred = np.uint8(tf.clip_by_value(rgb_pred*255., 0., 255.))
    
    return rgb_pred
    
def get_argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_weights', type=str, help='Path to the model weights.',
                        default='./h5/anvnet_ep300.h5')
    parser.add_argument('--path_dataset', type=str, help='Path to the dataset.',
                        default='./data')
    parser.add_argument('--path_save', type=str, help='Path to save results.',
                        default='./results')
    parser.add_argument('--height', type=int, help='Input height of the images.',
                        default=2816)
    parser.add_argument('--width', type=int, help='Input width of the images.',
                        default=4096)

    # parse configs
    return parser.parse_args()

if __name__=='__main__':

    args = get_argumets()

    HEIGHT, WIDTH = args.height, args.width
    folder_images = args.path_dataset
    folder_save = args.path_save
    
    
    model = get_model(shape=(HEIGHT,WIDTH), batch_size=1, resize_output=True)
    model.load_weights(args.path_weights)
    print(model.inputs)
    print(model.outputs)
    
    os.makedirs(folder_save, exist_ok=True)
    path_folders = glob(os.path.join(folder_images, '*'))
    
    for folder in tqdm(path_folders):
    
        name = os.path.split(folder)[-1]
    
        path_under = os.path.join(folder, '1.jpg')
        path_over  = os.path.join(folder, '2.jpg')
        
        if not (os.path.exists(path_under) and os.path.exists(path_over)):
            continue
            
        img_under = cv2.imread(path_under)
        
        y_under, uv_under = read_input_data(path_under)
        y_over, uv_over   = read_input_data(path_over)
        
        y_pred, uv_pred = model([y_under, uv_under, y_over, uv_over])
        bgr_pred = convert_prediction(y_pred, uv_pred)
        bgr_pred = cv2.resize(bgr_pred, img_under.shape[:2][::-1])
        
        cv2.imwrite(os.path.join(folder_save, name)+'.jpg', bgr_pred)