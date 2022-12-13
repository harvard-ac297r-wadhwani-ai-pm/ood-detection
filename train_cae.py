#!/usr/bin/env python3
import os
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras.backend as K
from models.autoencoder import Autoencoder, SSIMLoss
from data_utils.datasets import find_image_files, build_dataset

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

def parse_args():

    parser = argparse.ArgumentParser('Autoencoder')
    none_or_str = lambda x: None if x.lower() == 'none' else str(x)
    none_or_int = lambda x: None if x.lower() == 'none' else int(x)
    none_or_float = lambda x: None if x.lower() == 'none' else float(x)
    str2bool = lambda x: x.lower() in ('true', '1')
    ''' Dataset arguments '''
    parser.add_argument('--id-data-folder', type=str, default='data/bollworms-train/ID', help='folder with ID training data [%(default)s]')
    parser.add_argument('--ood-data-folder', type=none_or_str, default='data/bollworms-train/OOD', help='folder with OOD data (for plotting only) [%(default)s]')
    parser.add_argument('--val-size', type=float, default=0.2, help='fractional size of validation set [%(default)g]')
    parser.add_argument('--cache', action='store_true', help='cache all datasets in memory/on-disk')
    parser.add_argument('--image-shape', type=int, nargs='+', default=[256, 256, 3], help='input image shape [%(default)s]')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training and testing [%(default)i]')
    parser.add_argument('--shuffle-buffer', type=int, default=512, help='number of images in shuffle buffer [%(default)i]')
    parser.add_argument('--random-state', type=none_or_int, default=None, help='random_state for train_test_split method [%(default)s]')
    ''' Autoencoder arguments '''
    parser.add_argument('--name', type=none_or_str, default=None, help='name of model and log directory [%(default)s]')
    parser.add_argument('--init-filters', type=int, default=32, help='number of filters in first layer of encoder [%(default)i]')
    parser.add_argument('--filter-mult', type=int, default=2, help='filter multiplier [%(default)i]')
    parser.add_argument('--filter-mult-every', type=int, default=2, help='filter multiplier rate [%(default)i]')
    parser.add_argument('--max-encode-blocks', type=none_or_int, default=None, help='maximum number of encode blocks [%(default)s]')
    parser.add_argument('--kernel-size', type=int, default=3, help='kernel size for convolution layers [%(default)s]')
    parser.add_argument('--activation', type=str, choices=['relu', 'leakyrelu'], default='leakyrelu', help='activation function [%(default)s]')
    parser.add_argument('--negative-slope', type=float, default=0.2, help='hyperparameter for LeakyReLU activation function [%(default)s]')
    parser.add_argument('--batch-norm', action='store_true', help='use batch normalization layers')
    parser.add_argument('--kernel-regularizer', type=none_or_str, choices=[None, 'L1', 'L2'], default=None, help='kernel regularizer [%(default)s]')
    parser.add_argument('--kernel-initializer', type=str, default='he_normal', help='kernel initializer [%(default)s]')
    ''' Training arguments '''
    parser.add_argument('--loss', type=str, choices=['mse', 'mae', 'ssim'], default='mse', help='loss function [%(default)s]')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='optimizer [%(default)s]')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate [%(default)g]')
    parser.add_argument('--clipvalue', type=none_or_float, default=2.0, help='clip gradients below this value [%(default)s]')
    parser.add_argument('--method', type=str, choices=['greedy', 'end_to_end', 'fine_tune'], default='greedy', help='training method [%(default)s]')
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs [%(default)i]')
    parser.add_argument('--patience', type=int, default=25, help='patience parameter for early stopping [%(default)i]')
    parser.add_argument('--verbose', action='store_true', help='verbose output during training')
    return parser.parse_args()

def main(args):

    ## BUILD DATASETS

    # These keyword arguments will be shared by all image datasets
    ds_kwargs = {'cache': args.cache, 'image_shape': tuple(args.image_shape), 'batch_size': args.batch_size, 'shuffle_buffer': args.shuffle_buffer}

    # Build training and validation sets of ID images for autoencoder model
    train_id_df = find_image_files(os.path.abspath(os.path.expanduser(args.id_data_folder)))
    train_id_df, val_id_df = train_test_split(train_id_df, test_size=args.val_size, random_state=args.random_state)
    train_id_ds = build_dataset(train_id_df, augment=True, shuffle=True, **ds_kwargs)
    val_id_ds = build_dataset(val_id_df, augment=False, shuffle=False, **ds_kwargs)

    print('')

    ## BUILD MODEL

    if args.loss == 'mse':
        args.loss = tf.keras.losses.MeanSquaredError()
    elif args.loss == 'mae':
        args.loss = tf.keras.losses.MeanAbsoluteError()
    else:
        args.loss = SSIMLoss()

    if args.activation == 'relu':
        args.activation = tf.keras.layers.ReLU()
    else:
        args.activation = tf.keras.layers.LeakyReLU(alpha=args.negative_slope)

    if args.optimizer == 'adam':
        args.optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipvalue=args.clipvalue)
    else:
        args.optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, clipvalue=args.clipvalue)

    autoencoder = Autoencoder(
        name=args.name,
        init_filters=args.init_filters, 
        filter_mult=args.filter_mult, 
        filter_mult_every=args.filter_mult_every, 
        max_encode_blocks=args.max_encode_blocks,
        kernel_size=args.kernel_size, 
        activation=args.activation,
        batch_norm=args.batch_norm,
        kernel_regularizer=args.kernel_regularizer,
        kernel_initializer=args.kernel_initializer,
        image_shape=ds_kwargs['image_shape'])
    autoencoder.build((None, *ds_kwargs['image_shape']))
    autoencoder.compile(loss=args.loss, optimizer=args.optimizer)
    autoencoder.summary()

    ## TRAIN MODEL

    autoencoder.train(
        train_id_ds, 
        val_id_ds, 
        method=args.method, # ['greedy', 'end_to_end', 'fine_tune']
        epochs=args.epochs, 
        patience=args.patience, 
        savefigs=True, 
        resume_from=None, 
        verbose=args.verbose)

    ## PLOT METRICS

    train_ds_dict = {'Train ID': train_id_ds, 'Val ID': val_id_ds}

    if args.ood_data_folder is not None:
        train_ood_df = find_image_files(os.path.abspath(os.path.expanduser(args.ood_data_folder)))
        train_ood_ds = build_dataset(train_ood_df, augment=False, shuffle=False, **ds_kwargs)
        train_ds_dict['Train OOD'] = train_ood_ds

    layers = range(1, autoencoder.encode_blocks)
    train_loss_dicts = autoencoder.plot_loss(train_ds_dict, layers=layers, savefigs=True)
    train_ssim_dicts = autoencoder.plot_ssim(train_ds_dict, layers=layers, savefigs=True)

if __name__ == '__main__':

    args = parse_args()
    print(args, '\n')
    main(args)
