import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Common default values
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1024

def get_image_labels(image_files):
    '''
    Converts list of image filenames into a pandas dataframe.
    '''
    return pd.DataFrame(image_files, columns=['FILE'])


def load_image(image_file):
    '''
    Assumes image filename is a string tensor.
    '''
    image_file = image_file.numpy().decode('utf-8')
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32) / 255.0
    if image.shape[2] == 1:
        tf_z_exp = tf.expand_dims(image,axis=2)
        image = tf.repeat(tf_z_exp,repeat=3,axis=2)
        #image = tf.image.resize(image, [256,256])
        #image = cv2.merge((image.numpy(),image.numpy(),image.numpy()))
    else:
        image = tf.image.resize(image, [256,256])
    return image


def tf_load_image(image_file, image_shape):
    '''
    Wrapper around load_image routine that sets the shape property of the
    returned tensor. This step is unfortunately required if we want to load
    images by filename using the map routine in our tf.data model below.
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


def create_dataset(image_labels,
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
    # Create numpy array of image filenames
    image_files = image_labels['FILE'].to_numpy()

    # Build input dataset
    ds_inputs = tf.data.Dataset.from_tensor_slices(image_files)
    ds_inputs = ds_inputs.map(lambda image_file: tf_load_image(image_file, image_shape), num_parallel_calls=AUTOTUNE)
    if cache: ds_inputs = ds_inputs.cache(filename=cache_filename)
    if shuffle: ds_inputs = ds_inputs.shuffle(shuffle_buffer)

    # Build final dataset by zipping two copies of the inputs dataset.
    ds = tf.data.Dataset.zip((ds_inputs, ds_inputs))
    if augment: ds = ds.map(augment_images, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def build_train_dataset(
        id_folder,
        ec_folder=None,
        max_images=None,
        val_size=0.2,
        random_state=None,
        augment=[True, False], # [train, val]
        shuffle=[True, False], # [train, val]
        cache=[True, True], # [train, val]
        cache_filename=['', ''], # no filename -> RAM, filename -> disk
        image_shape=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        shuffle_buffer=SHUFFLE_BUFFER):
    '''
    Wrapper around create_dataset routine for generating the training and
    validation datasets. The dataset will consist of all in-distribution and
    edge-case images found in id_folder and ec_folder, respectively. The user
    can optionally ignore edge-case images by setting ec_folder=None.
    '''
    # Get list of image files in the ID set
    id_image_files = glob.glob(f'{id_folder}/*.jpg')
    if len(id_image_files) == 0:
        raise OSError(f'No ID images found in folder "{id_folder}"')
    print(f'Found {len(id_image_files)} ID images.')

    # Get list of image files in EC set if one was provided
    if ec_folder is not None:
        ec_image_files = glob.glob(f'{ec_folder}/*.jpg')
        print(f'Found {len(ec_image_files)} EC images.')
    else:
        ec_image_files = []

    # Read and process image labels
    trainval_df = get_image_labels(id_image_files + ec_image_files)

    # If the user specified a maximum number of images, first shuffle the dataframe
    # to mix ID and EC images together and then take the requested number from the top. 
    if max_images is not None:
        trainval_df = trainval_df.sample(frac=1).reset_index(drop=True).head(max_images)

    # Perform train/val split on full dataset
    train_df, val_df = train_test_split(trainval_df, test_size=val_size, random_state=random_state)

    # Create training and validation datasets
    train_ds = create_dataset(train_df,
                              augment=augment[0],
                              shuffle=shuffle[0],
                              cache=cache[0],
                              cache_filename=cache_filename[0],
                              image_shape=image_shape,
                              batch_size=batch_size,
                              shuffle_buffer=shuffle_buffer)

    val_ds = create_dataset(val_df,
                            augment=augment[1],
                            shuffle=shuffle[1],
                            cache=cache[1],
                            cache_filename=cache_filename[1],
                            image_shape=image_shape,
                            batch_size=batch_size,
                            shuffle_buffer=shuffle_buffer)

    print(f'Built train dataset with {len(train_df)} images.')
    print(f'Built validation dataset with {len(val_df)} images.')

    return (train_ds, train_df), (val_ds, val_df)

def build_test_dataset(
        ood_folder,
        external_ood=False,
        max_images=None,
        augment=False,
        shuffle=False,
        cache=True,
        cache_filename='', # no filename -> RAM, filename -> disk
        image_shape=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        shuffle_buffer=SHUFFLE_BUFFER):
    '''
    Wrapper around create_dataset routine for generating test dataset. The test
    dataset will consist of all out-of-distribution images found in ood_folder.
    '''

    # Get list of image files in the OOD set
    
    if external_ood == False:
        ood_image_files = glob.glob(f'{ood_folder}/*.jpg')
    else:
        ood_image_files = glob.glob(f'{ood_folder}/*.JPEG')
    
    if len(ood_image_files) == 0:
        raise OSError(f'No OOD images found in folder "{ood_folder}"')
    print(f'Found {len(ood_image_files)} OOD images.')

    # Read and process image labels
    test_df = get_image_labels(ood_image_files)

    # If the user specified a maximum number of images, take them off the top.
    if max_images is not None:
        test_df = test_df.head(max_images)

    # Create training and validation datasets
    test_ds = create_dataset(test_df,
                             augment=augment,
                             shuffle=shuffle,
                             cache=cache,
                             cache_filename=cache_filename,
                             image_shape=image_shape,
                             batch_size=batch_size,
                             shuffle_buffer=shuffle_buffer)

    print(f'Built test dataset with {len(test_df)} images.')

    return (test_ds, test_df)
