import glob
import numpy as np
import pandas as pd
import tensorflow as tf

# Establish default values
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1024

def find_image_files(data_folder, verbose=True):
    '''
    Returns pandas dataframe with image files found in data_folder
    '''
    image_files = []
    for filetype in ['*.jpg', '*.jpeg', '*.JPEG']:
        image_files.extend(glob.glob(f'{data_folder}/{filetype}'))

    if verbose:
        print(f'Found {len(image_files)} image files in folder "{data_folder}"')

    return pd.DataFrame(image_files, columns=['file'])


def load_image(image_file):
    '''
    Assumes image_file is a string tensor.
    '''
    image_file = image_file.numpy().decode('utf-8')
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SHAPE[:2])
    return image


def tf_load_image(image_file, image_shape):
    '''
    Wrapper around load_image routine that sets the shape property of the
    returned tensor. This step is required if we want to load images by
    filename using the map routine in our tf.data model below.
    '''
    [image] = tf.py_function(load_image, [image_file], [tf.float32])
    image.set_shape(image_shape)
    return image


def augment_images(inputs, targets):
    '''
    Randomly flips, crops, and adjusts the contrast and brightness of images.
    Works on batches of images but applies random augmentation on a per-image
    (not per-batch) basis.
    '''
    # These routines all seem to work best if converted to uint8 first.
    modified = tf.image.convert_image_dtype(inputs, tf.uint8)
    modified = tf.image.random_flip_left_right(modified)
    modified = tf.image.random_flip_up_down(modified)

    # Random zoom achived by upsampling and then random crop. Idea taken from
    # https://www.tensorflow.org/tutorials/generative/pix2pix
    modified = tf.image.resize(modified, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    modified = tf.image.random_crop(modified, size=IMAGE_SHAPE)

    # Trying these out as well, not sure if we need them all, may need more aggressive augmentation
    #modified = tf.image.random_jpeg_quality(modified, 90, 100)
    modified = tf.image.random_contrast(modified, 0.75, 1.25)
    modified = tf.image.random_brightness(modified, 0.25)
    #modified = tf.image.random_hue(modified, 0.02)

    # Convert back to float32 afer augmentation
    modified = tf.image.convert_image_dtype(modified, tf.float32)

    return (modified, modified)


def build_dataset(image_files, # pd.DataFrame, np.array, or list
                  augment=True,
                  shuffle=True,
                  cache=True,
                  cache_filename='', # no filename -> RAM, filename -> disk
                  image_shape=IMAGE_SHAPE,
                  batch_size=BATCH_SIZE,
                  shuffle_buffer=SHUFFLE_BUFFER):
    '''
    Produces a generic tf.data.Dataset for our autoencoder model. This dataset
    returns two identical copies of each image: one as the input and one as the
    output that the autoencoder is trying to reproduce through a bottleneck.
    '''
    # Convert image_files to numpy array
    if isinstance(image_files, pd.DataFrame):
        image_files = image_files['file'].to_numpy()
    else:
        image_files = np.asarray(image_files, dtype=object)

    # Build dataset of singular input images
    ds_inputs = tf.data.Dataset.from_tensor_slices(image_files)
    ds_inputs = ds_inputs.map(lambda image_file: tf_load_image(image_file, image_shape), num_parallel_calls=AUTOTUNE)
    if cache: ds_inputs = ds_inputs.cache(filename=cache_filename)
    if shuffle: ds_inputs = ds_inputs.shuffle(shuffle_buffer)

    # Build final dataset by zipping two copies of the inputs dataset
    ds = tf.data.Dataset.zip((ds_inputs, ds_inputs))
    if augment: ds = ds.map(augment_images, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
