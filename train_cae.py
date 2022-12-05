#!/usr/bin/env python3
import os
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras.backend as K
from models.autoencoder import Autoencoder, SSIMLoss
from data_utils.datasets import build_train_dataset, build_test_dataset

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

def parse_args():

    parser = argparse.ArgumentParser('Autoencoder')
    none_or_str = lambda x: None if x.lower() == 'none' else str(x)
    none_or_int = lambda x: None if x.lower() == 'none' else int(x)
    none_or_float = lambda x: None if x.lower() == 'none' else float(x)
    str2bool = lambda x: x.lower() in ('true', '1')
    parser.add_argument('--data-folder', type=str, default='data', help='folder with train/test data [%(default)s]')
    parser.add_argument('--no-edge-cases', action='store_true', help='include edge cases in training set')
    parser.add_argument('--max-train-images', type=none_or_int, default=None, help='maximum number of training images [%(default)s]')
    parser.add_argument('--max-test-images', type=none_or_int, default=None, help='maximum number of testing images [%(default)s]')
    parser.add_argument('--val-size', type=float, default=0.2, help='fractional size of validation set [%(default)g]')
    parser.add_argument('--cache', action='store_true', help='cache all datasets in memory/on-disk')
    parser.add_argument('--image-shape', type=int, nargs='+', default=[256, 256, 3], help='input image shape [%(default)s]')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training and testing [%(default)i]')
    parser.add_argument('--shuffle-buffer', type=int, default=2056, help='number of images in shuffle buffer [%(default)i]')
    parser.add_argument('--name', type=none_or_str, default=None, help='name of model and log directory [%(default)s]')
    parser.add_argument('--init-filters', type=int, default=32, help='number of filters in first layer of encoder [%(default)i]')
    parser.add_argument('--filter-mult', type=int, default=2, help='filter multiplier [%(default)i]')
    parser.add_argument('--filter-mult-every', type=int, default=2, help='filter multiplier rate [%(default)i]')
    parser.add_argument('--max-encode-blocks', type=none_or_int, default=None, help='maximum number of encode blocks [%(default)s]')
    parser.add_argument('--kernel-size', type=int, default=3, help='kernel size for convolution layers [%(default)s]')
    parser.add_argument('--activation', type=str, choices=['relu', 'leakyrelu'], default='relu', help='activation function [%(default)s]')
    parser.add_argument('--negative-slope', type=float, default=0.2, help='hyperparameter for LeakyReLU activation function [%(default)s]')
    parser.add_argument('--batch-norm', action='store_true', help='use batch normalization layers')
    parser.add_argument('--kernel-regularizer', type=none_or_str, choices=[None, 'L1', 'L2'], default=None, help='kernel regularizer [%(default)s]')
    parser.add_argument('--kernel-initializer', type=str, default='he_normal', help='kernel initializer [%(default)s]')
    parser.add_argument('--loss', type=str, choices=['mse', 'mae', 'ssim'], default='mse', help='loss function [%(default)s]')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='optimizer [%(default)s]')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate [%(default)g]')
    parser.add_argument('--clipvalue', type=none_or_float, default=None, help='clip gradients below this value [%(default)s]')
    parser.add_argument('--method', type=str, choices=['greedy', 'end_to_end', 'fine_tune'], default='greedy', help='training method [%(default)s]')
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs [%(default)i]')
    parser.add_argument('--patience', type=int, default=25, help='patience parameter for early stopping [%(default)i]')
    parser.add_argument('--verbose', action='store_true', help='verbose output during training')
    return parser.parse_args()

def main(args):

    ## BUILD DATASETS

    args.data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    args.image_shape = tuple(args.image_shape)

    (train_ds, train_df), (val_ds, val_df) = build_train_dataset(
        id_folder=f'{args.data_folder}/ID',
        ec_folder=(None if args.no_edge_cases else f'{args.data_folder}/EC'), # can be None
        max_images=args.max_train_images,
        val_size=args.val_size,
        augment=[True, False], # [train, val]
        shuffle=[True, False], # [train, val]
        cache=[args.cache, args.cache], # [train, val]
        cache_filename=['', ''], # no filename -> RAM, filename -> disk
        image_shape=args.image_shape,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer)

    test_ds, test_df = build_test_dataset(
        ood_folder=f'{args.data_folder}/OOD', 
        max_images=args.max_test_images,
        augment=False,
        shuffle=False,
        cache=args.cache,
        cache_filename='', # no filename -> RAM, filename -> disk
        image_shape=args.image_shape,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer)

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
        kernel_initializer=args.kernel_initializer)
    autoencoder.build((None, *args.image_shape))
    autoencoder.compile(loss=args.loss, optimizer=args.optimizer)
    autoencoder.summary()

    ## TRAIN MODEL

    autoencoder.train(
        train_ds, 
        val_ds, 
        method=args.method, # ['greedy', 'end_to_end', 'fine_tune']
        epochs=args.epochs, 
        patience=args.patience, 
        savefigs=True, 
        resume_from=None, 
        verbose=args.verbose)

    ## PLOT METRICS

    train_losses, val_losses, test_losses = autoencoder.plot_loss(
        train_ds, 
        val_ds, 
        test_ds, 
        layers=range(1, autoencoder.encode_blocks),
        savefigs=True)

    train_ssims, val_ssims, test_ssims = autoencoder.plot_ssim(
        train_ds, 
        val_ds, 
        test_ds, 
        layers=range(1, autoencoder.encode_blocks),
        savefigs=True)

if __name__ == '__main__':

    args = parse_args()
    print(args, '\n')
    main(args)
