import numpy as np

import re
from PIL import ImageOps, ImageEnhance, ImageFilter, Image, ImageDraw
import random
from dataclasses import dataclass
from typing import Union
from random import randint
import PIL
@dataclass
class MinMax:
    min: Union[float, int]
    max: Union[float, int]


@dataclass
class MinMaxVals:
    shear: MinMax = MinMax(.0, .2)
    translate: MinMax = MinMax(0, 10)  # different from uniaug: MinMax(0,14.4)
    rotate: MinMax = MinMax(0, 30)
    enhancer: MinMax = MinMax(0.75,1.25 )
    cutout: MinMax = MinMax(.0, .2)
    color: MinMax = MinMax(.75, 1.25) 
    randomerase: MinMax = MinMax(.0, .1)
    videonoise: MinMax = MinMax(.0,0.01)
'''    
#alltrivialaug
min_max_vals = MinMaxVals(
            shear=MinMax(.0, .1),
            translate=MinMax(0, 10),
            rotate=MinMax(0, 10),
            enhancer=MinMax(.5, 1.5),
            cutout=MinMax(.0,0.1)


            
        )
'''
#alltrivialaug2
min_max_vals = MinMaxVals(
            shear=MinMax(0, 0.10),
            translate=MinMax(0, 10),
            rotate=MinMax(0, 10),
            enhancer=MinMax(0.75, 1.25),
            cutout=MinMax(.0,0.10),
            randomerase = MinMax(0, 0.1),




        )

def _enhancer_impl(enhancer, minimum=None, maximum=None):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        mini = min_max_vals.enhancer.min if minimum is None else minimum
        maxi = min_max_vals.enhancer.max if maximum is None else maximum
        v = float_parameter(level, maxi - mini) + mini  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl
'''
def _color_impl(enhancer, minimum=None, maximum=None):
        """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl_color(pil_img, level):
        
        mini = min_max_vals.color.min if minimum is None else minimum
        maxi = min_max_vals.color.max if maximum is None else maximum
        v = float_parameter(level, maxi - mini) + mini  # going to 0 just destroys it
        print("v")
        print(v)
        return enhancer(pil_img).enhance(v)

     return impl_color
'''
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
    return float((level) * (maxval-minval) / (PARAMETER_MAX) +minval)
def int_parameter2(level, minval,maxval):
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
    return int((level) * (maxval-minval) / (PARAMETER_MAX) +minval)

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

class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def __repr__(self):
        return '<' + self.name + '>'

    def pil_transformer(self, probability, level):
        def return_function(im):
            if random.random() < probability:
                im = self.xform(im, level)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

def _shear_y_impl(pil_img, level):
    """Applies PIL ShearY to `pil_img`.

  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random()>0.5:
        level = -level

    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)

def _translate_y_impl(pil_img, level):
    """Applies PIL TranslateY to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random()>0.5:
        level = -level

    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)

def _shear_x_impl(pil_img, level):
    """Applies PIL ShearX to `pil_img`.

  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random()>0.5:
        level = -level
    #print(level)

    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)


def _translate_x_impl(pil_img, level):
    """Applies PIL TranslateX to `pil_img`.

  Translate the image in the horizontal direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random()>0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)

def _rotate_impl(pil_img, level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level,min_max_vals.rotate.max)
    if random.random()>0.5:
        degrees = -degrees
    #print(degrees)

    return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)

flip_lr = TransformT(
    'FlipLR',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))

brightness = TransformT('Brightness', _enhancer_impl(
    ImageEnhance.Brightness))

def CutoutDefault(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20

    #print("v")
    #print(v)
    if v <= 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

def RandomeraseDefault(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    #print("v")
    #print(v)
    v=7


    if v <= 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    array = np.array([[[randint(0, 255), randint(0, 255), randint(0, 255)]] for i in range(v*v)])
    array = np.reshape(array.astype('uint8'), (v, v, 3))
    randimg = Image.fromarray(np.uint8(array.astype('uint8')))


    #img = img.copy(randimg,(x0,y0))
    img.paste(randimg,(x0,y0))
    #ImageDraw.Draw(img).rectangle(xy, color)
    return img


cutout = TransformT('Cutout',
                    lambda img, l: CutoutDefault(img, int_parameter(l, img.size[0] * min_max_vals.cutout.max)))
randomerase = TransformT('Randomerase',
                    lambda img, l: RandomeraseDefault(img, int_parameter(l, img.size[0] * min_max_vals.randomerase.max)))
def _videonoise_impl(data_numpy,level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    #level = float_parameter(level, min_max_vals.videonoise.max)
    #print(level)
    H,W,C = data_numpy.shape
    #guassian_part_var = level * level
    #poisson_part_coeff = level*0.05
    guassian_part_var = 0.01 * 0.01
    poisson_part_coeff = 0.0005
    data_mean = np.mean(data_numpy)
    guassian_poisson_var = guassian_part_var + poisson_part_coeff * data_mean
    data_numpy1 = data_numpy + np.random.randn(H, W, C) * (guassian_poisson_var ** 0.5)

    return data_numpy1
videonoise =  TransformT('VideoNoise', _videonoise_impl)
color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
#all_v2
ALL_SPATIAL_TRANSFORMS = [
        shear_x,
        shear_y,
        translate_x,
        translate_y,
        rotate,
        brightness,
        flip_lr,
        cutout,
        randomerase,
        videonoise,
        color
        #compression
    ]

PARAMETER_MAX = 4

class TrivialAugment:
    def __init__(self):

        self.op = random.choices(ALL_SPATIAL_TRANSFORMS, k=1)[0]

        print(self.op)
        self.level = random.randint(0, PARAMETER_MAX)
        print(self.level)
    def __call__(self, data_numpy):

        if self.op == videonoise:
            return self.op.pil_transformer(1., self.level)(data_numpy)
        else:
            data_numpy = data_numpy * 255
            data_numpy = data_numpy.astype(np.uint8)
            img = PIL.Image.fromarray(data_numpy)
            img = self.op.pil_transformer(1., self.level)(img)
            data_numpy = np.asarray(img)
            data_numpy = data_numpy / 255
        return data_numpy
