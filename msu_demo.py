#!/usr/bin/env python

import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from imageio import imsave
from tqdm import tqdm

from dh_segment.inference import LoadedModel
from dh_segment.post_processing import binarization

project_dir = 'msu_demo'

def page_make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    mask = binarization.thresholding(probs, threshold)
    mask = binarization.cleaning_binary(mask, kernel_size=5)
    return mask


def get_classes():
    f = open(project_dir + "/input/classes.txt", "r")
    classes = []
    for line in f:
        cl = []
        for nbr in line.split():
            cl.append(int(nbr))
        classes.append(cl)

    return classes


if __name__ == '__main__':

    # If the model has been trained load the model, otherwise use the given model
    model_dir = project_dir + '/page_model/export'
    if not os.path.exists(model_dir):
        model_dir = project_dir + '/model/'

    input_files = glob(project_dir + '/input/test_a1/images/*')
    output_dir = project_dir + '/output'

    os.makedirs(output_dir, exist_ok=True)

    with tf.Session():  # Start a tensorflow session
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')

        for filename in tqdm(input_files, desc='Processed files'):
            # For each image, predict each pixel's label
            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]
            original_shape = prediction_outputs['original_shape']
            classes = get_classes()

            img = Image.open(filename, 'r')
            pixels = img.load()

            # Iterate over all classes
            for p, cl in enumerate(classes):
                if (p==0):
                    continue
                prob = probs[:, :, p]  # Take only class '1' (class 0 is the background, class 1 is the page)
                prob = prob / np.max(prob)  # Normalize to be in [0, 1]

                # Binarize the predictions
                page_bin = page_make_binary_mask(prob)

                # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
                bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                                          tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

                # Load image

                # Get class colors
                cr, cg, cb = cl[0:3]
                print("Color: ({0}, {1}, {2})".format(cr, cg, cb))

                # Mark each masked pixel with a transparent color
                for i, row in enumerate(bin_upscaled):
                    for j, col in enumerate(row):
                        # If pixel belongs to a class
                        if col == 1:
                            r, g, b = img.getpixel((j, i))
                            # Make class colors transparent
                            nr = int((cr + r) / 2)
                            ng = int((cg + g) / 2)
                            nb = int((cb + b) / 2)
                            pixels[j, i] = (nr, ng, nb)
               
            # Save output
            basename = os.path.basename(filename).split('.')[0]
            imsave(os.path.join(output_dir, '{}_marked.jpg'.format(basename)), img)
