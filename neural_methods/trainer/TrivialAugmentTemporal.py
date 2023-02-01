import numpy as np
import math
import re
from PIL import ImageOps, ImageEnhance, ImageFilter, Image, ImageDraw
import random
from dataclasses import dataclass
from typing import Union
from random import randint
import PIL
from neural_methods.trainer.TrivialAugment import *
@dataclass
class MinMax:
    min: Union[float, int]
    max: Union[float, int]


@dataclass
class MinMaxVals:
    shear: MinMax = MinMax(.0, .2)
    translate: MinMax = MinMax(0, 10)  # different from uniaug: MinMax(0,14.4)
    rotate: MinMax = MinMax(0, 30)
    gaussianlabel: MinMax = MinMax(0, 0.75)
    labelwarping: MinMax = MinMax(.5, 1.)
    enhancer: MinMax = MinMax(0.75, 1.25)
    cutout: MinMax = MinMax(.0, .2)
    scaling: MinMax = MinMax(0.75, 1.25)
    lowfreqnoisemagnitude: MinMax = MinMax(0,0.2)
    lowfreqnoiseomega: MinMax = MinMax(0,0.5)
    
'''
#allbatchtrivialaug
min_max_vals = MinMaxVals(
    shear=MinMax(.0, .1),
    translate=MinMax(0, 10),
    rotate=MinMax(0, 10),
    gaussianlabel=MinMax(.5, 0.8),
    labelwarping=MinMax(.5, 1.),
    enhancer=MinMax(.5, 1.5),
    cutout=MinMax(.0, 0.1)

)
'''
min_max_vals = MinMaxVals(
    
    gaussianlabel=MinMax(0, 0.5),
    labelwarping=MinMax(0, 1),
    scaling = MinMax(0.75,1.25),
    lowfreqnoisemagnitude= MinMax(0.01,0.5),
    lowfreqnoiseomega=MinMax(0.01,0.2),

)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX

def float_parameter2(level, minval,maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    #print(level)
    #print(maxval-minval)
    #print(1+PARAMETER_MAX)
    return float((level) * (maxval-minval) / (PARAMETER_MAX+1) +minval)

def float_parameter3(level, minval,maxval):
    return float((level) * (maxval-minval) / (PARAMETER_MAX) +minval)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)
def _enhancer_impl(enhancer, minimum=None, maximum=None):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        mini = min_max_vals.enhancer.min if minimum is None else minimum
        maxi = min_max_vals.enhancer.max if maximum is None else maximum
        v = float_parameter(level, maxi - mini) + mini  # going to 0 just destroys it
        #print("v")
        #print(v)
        return enhancer(pil_img).enhance(v)

    return impl

class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, video,label):
        return self.f(video,label)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def __repr__(self):
        return '<' + self.name + '>'

    def pil_transformer_temporal(self, probability, level):
        def return_function(video,label):
            if random.random() < probability:
                im = self.xform(video,label, level)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

def _reverse_impl(data_numpy,label_numpy,level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    '''
   print(data_numpy.shape)
    print(label_numpy.shape)
    data_numpy = np.flip(data_numpy[0,:,:,:,:], axis=0)
    data_numpy = np.flip(data_numpy[1,:,:,:,:], axis=0)
    data_numpy = np.flip(data_numpy[2,:,:,:,:], axis=0)
    data_numpy = np.flip(data_numpy[3,:,:,:,:], axis=0)
    label_numpy = np.flip(label_numpy[0,:], axis=0)
    label_numpy = np.flip(label_numpy[1,:], axis=0)
    label_numpy = np.flip(label_numpy[2,:], axis=0)
    label_numpy = np.flip(label_numpy[3,:], axis=0)
    '''
    data_numpy = np.flip(data_numpy, axis=1)
    label_numpy = np.flip(label_numpy, axis=1)
    return data_numpy, label_numpy
reverse =  TransformT('Reverse', _reverse_impl)



def _gaussianlabel_impl(data_numpy,label_numpy,level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    #var = float_parameter(level, min_max_vals.gaussianlabel.max)
    #print(var)
    '''
    if random.random () >0.5:
        #var = 0.5
        var = float_parameter2(level, min_max_vals.gaussianlabel.min,min_max_vals.gaussianlabel.max)
    else:
        var = 0
    '''
    var = 0.5
    label_numpy = label_numpy+(var ** 0.5) * np.random.randn(4,180)
    

    return data_numpy, label_numpy
gaussianlabel =  TransformT('Gaussianlabel', _gaussianlabel_impl)


def magnitude_warp( x, sigma, knot=4):
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def _labelwarping_impl(data_numpy,label_numpy,level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    # print("ls")
    # print(label_numpy.shape)

    #sigma = float_parameter2(level, min_max_vals.labelwarping.min,min_max_vals.labelwarping.max)
    sigma = 1
    #print(sigma)
    label_numpy = magnitude_warp(label_numpy.reshape(1,720,1),sigma).reshape(4,180)


    return data_numpy, label_numpy
labelwarping =  TransformT('LabelWarping', _labelwarping_impl)
def _scaling_impl(data_numpy,label_numpy,level):
    scale = float_parameter2(level, min_max_vals.scaling.min,min_max_vals.scaling.max)
    #scale =1 
    label_numpy = label_numpy*scale
    return data_numpy, label_numpy
scaling = TransformT('Scaling', _scaling_impl)

def _lowfreqnoise_impl(data_numpy,label_numpy,level):
    magnitudescale = min_max_vals.lowfreqnoisemagnitude.max-float_parameter2(level, min_max_vals.lowfreqnoisemagnitude.min,min_max_vals.lowfreqnoisemagnitude.max)
    omegascale = float_parameter2(level, min_max_vals.lowfreqnoiseomega.min,min_max_vals.lowfreqnoiseomega.max)
    #print(magnitudescale)
    #print(omegascale)
    func = np.zeros(720)
    omega = 2*math.pi*omegascale/30
    for i in range(720):
        func[i] =magnitudescale*np.sin(omega*i)
    
    label_numpy = label_numpy+func.reshape(4,180)
    return data_numpy, label_numpy

lowfreqnoise = TransformT('Lowfreqnoie', _lowfreqnoise_impl)
#all_v1
ALL_TEMPORAL_TRANSFORMS = [
        #reverse,
        # upsample,
        # downsample,
        gaussianlabel,
        labelwarping,
        scaling,
        lowfreqnoise,
    ]
POOL1 = [
        
        
        translate_y,
        shear_x,
        translate_x,
        videonoise,
        rotate,
        shear_y
        

        
    ]

POOL2_T = [
        
        lowfreqnoise,
        gaussianlabel
        ]
Combsvdshx_trxtry=[[videonoise,shear_x],[translate_x, translate_y]]
Combsincludesallop=[[videonoise,shear_x],[translate_x, translate_y],[translate_x,rotate],[translate_y,gaussianlabel],[shear_x,shear_y],[shear_x, lowfreqnoise]]
Combs=[[videonoise,shear_x],[translate_x, translate_y],[translate_x,rotate],[translate_y,gaussianlabel],[videonoise,translate_y],[shear_x,shear_y],[shear_x,gaussianlabel],[translate_x,shear_y],[translate_y,translate_x],[translate_x,shear_x],[rotate,translate_y],[shear_x, lowfreqnoise],[translate_y, videonoise],[videonoise,shear_x]]
Combs2=[[translate_x,gaussianlabel],[translate_y, videonoise],[shear_x, lowfreqnoise],[rotate,translate_y],[translate_x,shear_x],[translate_y,translate_x],[translate_x,shear_y],[shear_x,gaussianlabel],[shear_x,shear_y] ,[videonoise,translate_y],[translate_y,gaussianlabel],[translate_x,rotate],[translate_x, translate_y],[videonoise,shear_x]]
PARAMETER_MAX = 4

class TrivialAugmentTemporal:
    def __init__(self):




        self.op1 = translate_x
        self.op2 = rotate
        self.op3 = videonoise
        self.op4 = shear_x
        self.op5 = shear_y
        self.op6 = translate_y
        self.op7 = lowfreqnoise
        self.op8 = gaussianlabel


        self.level1 = random.randint(0, PARAMETER_MAX)
        self.level2 = random.randint(0, PARAMETER_MAX)
        self.level3 = random.randint(0, PARAMETER_MAX)
        self.level4 = random.randint(0, PARAMETER_MAX)
        self.level5 = random.randint(0, PARAMETER_MAX)
        self.level6 = random.randint(0, PARAMETER_MAX)
        self.level7 = random.randint(0, PARAMETER_MAX)
        self.level8 = random.randint(0, PARAMETER_MAX)
        #print(self.level)
        self.prob1 = random.random()
        self.prob2 = random.random()
        self.prob3 = random.random()
        self.prob4 = random.random()
        self.prob5 = random.random()
        self.prob6 = random.random()
        self.prob7 = random.random()
        self.prob8 = random.random()
        #print((1/(len(POOL1)+1)))


    def __call__(self, data_numpy,batch_label):

        N, D, C, H, W = data_numpy.shape
        NL,LL = batch_label.shape
        dataperbatch = []
        labelperbatch = []



        if self.prob1>0.5:
            if self.op1 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op1.pil_transformer_temporal(1., self.level1)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op1 == videonoise:
                            datapersample.append( self.op1.pil_transformer(1., self.level1)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op1.pil_transformer(1., self.level1)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)

        if self.prob2>0.5:
            dataperbatch = []
            labelperbatch = []

            if self.op2 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op2.pil_transformer_temporal(1., self.level2)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op2 == videonoise:
                            datapersample.append( self.op2.pil_transformer(1., self.level2)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op2.pil_transformer(1., self.level2)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)

        if self.prob3>0.5:
            dataperbatch = []
            labelperbatch = []

            if self.op3 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op3.pil_transformer_temporal(1., self.level3)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op3 == videonoise:
                            datapersample.append( self.op3.pil_transformer(1., self.level3)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op3.pil_transformer(1., self.level3)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)

        if self.prob4>0.5:
            dataperbatch = []
            labelperbatch = []

            if self.op4 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op4.pil_transformer_temporal(1., self.level4)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op4 == videonoise:
                            datapersample.append( self.op4.pil_transformer(1., self.level4)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op4.pil_transformer(1., self.level4)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)

        if self.prob5>0.5:
            dataperbatch = []
            labelperbatch = []

            if self.op5 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op5.pil_transformer_temporal(1., self.level5)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op5 == videonoise:
                            datapersample.append( self.op5.pil_transformer(1., self.level5)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op5.pil_transformer(1., self.level5)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)

        if self.prob6>0.5:
            dataperbatch = []
            labelperbatch = []

            if self.op6 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op6.pil_transformer_temporal(1., self.level6)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op6 == videonoise:
                            datapersample.append( self.op6.pil_transformer(1., self.level6)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op6.pil_transformer(1., self.level6)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)

        if self.prob7>0.5:
            dataperbatch = []
            labelperbatch = []

            if self.op7 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op7.pil_transformer_temporal(1., self.level7)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op7 == videonoise:
                            datapersample.append( self.op7.pil_transformer(1., self.level7)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op7.pil_transformer(1., self.level7)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)

        if self.prob8>0.5:
            dataperbatch = []
            labelperbatch = []

            if self.op8 in ALL_TEMPORAL_TRANSFORMS:

                data_numpy,batch_label =self.op8.pil_transformer_temporal(1., self.level8)(data_numpy,batch_label)



            else:


                for sample_idx in range(N):
                    datapersample = []
                    for frame_idx in range(D):
                        if self.op8 == videonoise:
                            datapersample.append( self.op8.pil_transformer(1., self.level8)(data_numpy[sample_idx,frame_idx,:,:,:]))
                        else:

                            data_numpy_frame = data_numpy[sample_idx,frame_idx,:,:,:] * 255
                            data_numpy_frame = data_numpy_frame.astype(np.uint8)
                            img = PIL.Image.fromarray(data_numpy_frame)
                            img = self.op8.pil_transformer(1., self.level8)(img)
                            data_numpy_frame = np.asarray(img)
                            data_numpy_frame = data_numpy_frame / 255
                            datapersample.append(data_numpy_frame)
                    dataperbatch.append(datapersample)
                data_numpy = np.array(dataperbatch)










        return data_numpy, batch_label, self.op1, self.level1
