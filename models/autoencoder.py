import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Flatten, Dense, Reshape, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, LeakyReLU, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow.keras.backend as K

# Establish default values
IMAGE_SHAPE = (256, 256, 3)

##
## CUSTOM AUTOENCODER MODEL
##

class EncodeBlock(Layer):
    '''
    Basic building block for encoder model.
    '''
    def __init__(self, 
                 name='encode_block', 
                 filters=32, 
                 kernel_size=3, 
                 activation=ReLU(), 
                 batch_norm=False,
                 kernel_regularizer=None,
                 kernel_initializer='he_normal'):

        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        self.conv1 = Conv2D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        if batch_norm: self.batchnorm1 = BatchNormalization()
        self.act1 = activation
        self.conv2 = Conv2D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        if batch_norm: self.batchnorm2 = BatchNormalization()
        self.act2 = activation
        self.maxpool = MaxPooling2D((2, 2))

    def call(self, x, training=None):
        x = self.conv1(x)
        if self.batch_norm: x = self.batchnorm1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        if self.batch_norm: x = self.batchnorm2(x, training=training)
        x = self.act2(x)
        x = self.maxpool(x)
        return x


class DecodeBlock(Layer):
    '''
    Basic building block for decoder model.
    '''
    def __init__(self, 
                 name='decode_block', 
                 filters=32, 
                 kernel_size=3, 
                 activation=ReLU(), 
                 batch_norm=False,
                 kernel_regularizer=None,
                 kernel_initializer='he_normal',
                 is_output=False,
                 image_shape=IMAGE_SHAPE):

        super().__init__(name=name)
        self.filters = [filters, image_shape[-1] if is_output else filters]
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.is_output = is_output

        self.upsample1 = UpSampling2D((2, 2))
        self.conv1 = Conv2D(self.filters[0], kernel_size, strides=1, padding='same', kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        if batch_norm: self.batchnorm1 = BatchNormalization()
        self.act1 = activation
        self.conv2 = Conv2D(self.filters[1], kernel_size, strides=1, padding='same', kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        if batch_norm and not is_output: self.batchnorm2 = BatchNormalization()
        self.act2 = Activation('sigmoid') if is_output else activation

    def call(self, x, training=None):
        x = self.upsample1(x)
        x = self.conv1(x)
        if self.batch_norm: x = self.batchnorm1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        if self.batch_norm and not self.is_output: x = self.batchnorm2(x, training=training)
        x = self.act2(x)
        return x


class Autoencoder(Model):
    '''
    Custom autoencoder model class.
    '''
    def __init__(self, **kwargs):

        super().__init__()

        # Dictionary of default values for supported keyword arguments.
        default_kwargs = {
            'name': None,
            'image_shape': IMAGE_SHAPE, 
            'init_filters': 32, 
            'filter_mult': 2,
            'filter_mult_every': 2, 
            'max_encode_blocks': None,
            'kernel_size': 3, 
            'activation': ReLU(),
            'batch_norm': False,
            'kernel_regularizer': None,
            'kernel_initializer': 'he_normal'
            }

        # Store each supported keyword argument as a model attribute, using the
        # default values above wherever necessary. We need to be careful here
        # in our handling of protected attributes from the Model parent class.
        protected_attributes = ['name']
        for key, val in default_kwargs.items():
            if key in protected_attributes:
                setattr(self, '_' + key, kwargs.get(key, val))
            else:
                setattr(self, key, kwargs.get(key, val))

        # Generate a timestamp to serve as the model name if None was provided.
        # Also store the model configuration so we can recreate it later.
        self._name = self._name if self._name is not None else time.strftime("%Y%m%d_%H%M%S")
        self.config = {key: getattr(self, key) for key in default_kwargs.keys()}
        self.folder = f'logs/{self._name}'

        # Infer the shrinkage factor of each encode block 
        self.shrinkage_factor = int(16 / EncodeBlock()(tf.zeros([1, 16, 16, 1])).shape[1])

        # Compute the number of encode blocks that we can support, then compute
        # the number of filters in each encode block
        self.encode_blocks = int(np.log(np.min(self.image_shape[:2])) // np.log(self.shrinkage_factor))
        if self.max_encode_blocks is not None: 
            self.encode_blocks = np.min([self.encode_blocks, self.max_encode_blocks])
        self.encode_filters = [int(self.init_filters * self.filter_mult**np.floor(block_num / self.filter_mult_every)) for block_num in range(self.encode_blocks)]

        # Build full encoder network
        self.encoder = Sequential(name='encoder')
        self.encoder.add(Input(shape=self.image_shape))
        for block_num in range(self.encode_blocks):
            self.encoder.add(
                EncodeBlock(
                    name=f'encode_{block_num}', 
                    filters=self.encode_filters[block_num], 
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    batch_norm=self.batch_norm,
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_initializer=self.kernel_initializer
                )
            )

        # The decoder will have the same structure, but in reverse
        self.decode_blocks = self.encode_blocks
        self.decode_filters = [self.encode_filters[0], *self.encode_filters[:-1]]        

        # Build full decoder network  
        self.decoder = Sequential(name='decoder')
        self.decoder.add(Input(shape=self.encoder.layers[-1].output_shape[1:]))
        for block_num in reversed(range(self.decode_blocks)):
            self.decoder.add(
                DecodeBlock(
                    name=f'decode_{block_num}',
                    filters=self.decode_filters[block_num],
                    kernel_size=self.kernel_size, 
                    activation=self.activation,
                    batch_norm=self.batch_norm,
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_initializer=self.kernel_initializer,
                    is_output=(block_num == 0),
                    image_shape=self.image_shape
                )
            )

        # Initialize all staged models
        self.build_staged_models()


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def summary(self):
        '''
        Overloads summary method to print more useful information.
        '''
        self.encoder.summary()
        self.decoder.summary()
        print(f'Latent dimensions: {[np.prod(layer.output_shape[1:]) for layer in self.encoder.layers]}')
        print(f'Image compression: {[float(f"{np.prod(layer.output_shape[1:])/np.prod(self.image_shape):.3g}") for layer in self.encoder.layers]}')

    
    def save_config(self):
        '''
        Pickle model configuration so we can recreate this exact model later.
        '''
        # Create log directory for this model if it does not already exist
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Add compiled attributes to model configuration if they exist
        try:
            for compiled_attr in ['loss']:
                self.config[compiled_attr] = getattr(self, compiled_attr)
        except AttributeError:
            print('Model configuration saved without compiled attributes')

        # Pickle model configuration
        config_file = f'{self.folder}/{self._name}_config.pkl'
        with open(config_file, 'wb') as f:
            pickle.dump(self.config, f)

    
    @classmethod
    def from_config(cls, name):
        '''
        Builds Autoencoder model from previously saved configuration file.
        This does not restore the model weights, it simply generates an
        instance of the Autoencoder class with the same input arguments, which
        is a precursor to restoring the model weights.
        '''
        # Attempt to read in model configuration
        config_file = f'logs/{name}/{name}_config.pkl'
        try:
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
        except:
            raise OSError(f'config file not found: {config_file}')
        
        # Build model from configuration
        model = cls(**config)

        # Add compiled attributes to model if they exist
        try:
            for compiled_attr in ['loss']:
                setattr(model, compiled_attr, config[compiled_attr])
        except AttributeError:
            print('Model configuration loaded without compiled attributes')

        return model


    @classmethod
    def load_model(cls, name):
        '''
        Returns existing Autoencoder model with the specified name.
        '''
        model = cls.from_config(name)
        model.build((None, *model.config['image_shape']))
        model.load_weights(f'{model.folder}/{model._name}.h5')
        return model
        
    
    ##
    ## TRAINING ROUTINES
    ##

    def train(self, 
              train_ds,
              val_ds,
              method='greedy',
              epochs=150,
              patience=20,
              savefigs=True,
              resume_from=None,
              verbose=1):
        '''
        Trains autoencoder model using the specified method. This is a wrapper that
        provides a common interface for each of our custom fitting routines below.
        '''
        if method in ['greedy']: 
            self.fit_in_stages(train_ds, val_ds, epochs, patience, savefigs, resume_from, verbose)

        elif method in ['end_to_end']:
            self.fit_end_to_end(train_ds, val_ds, epochs, patience, savefigs, verbose)

        elif method in ['fine_tune']:
            self.fine_tune(train_ds, val_ds, epochs, patience, savefigs, verbose)

        else:
            raise ValueError(f'unknown method: {method}')


    def build_staged_models(self):
        '''
        Initialize staged models for greedy layer-wise training process
        '''
        self.staged_models = []
        for i in range(self.encode_blocks):
            self.staged_models.append(
                Sequential([*self.encoder.layers[:(i+1)], *self.decoder.layers[-(i+1):]], name=f'{self._name}_{i}')
            )
            self.staged_models[i].build((None, *self.image_shape))


    def fit_in_stages(self, 
                      train_ds, 
                      val_ds, 
                      epochs=150,
                      patience=10, 
                      savefigs=True,
                      resume_from=None,
                      verbose=1):
        '''
        Performs greedy layer-wise training of autoencoder model. Most of the
        code below is generating logs, checkpoints, and plots to allow the user
        to monitor the training process.
        '''
        # Create log directory for this model if it does not already exist
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Save the model configuration in log directory
        self.save_config()

        # Helper routine for printing messages to the console during training
        print_banner = lambda msg, char, num=30: print('\n' + char*num + f' {msg} ' + char*num + '\n')
        if verbose: print_banner('BEGINNING GREEDY TRAINING PROCESS', '=', 17)

        # Begin loop over training stages
        for i in range(self.encode_blocks):

            # Freeze weights in every layer except for last encode block and first decode block
            for j in range(i):
                self.staged_models[i].layers[j].trainable = False
                self.staged_models[i].layers[-(j+1)].trainable = False

            # If resuming from previous training run that crashed
            if resume_from is not None:
                if i < resume_from:
                    continue
                elif i == resume_from:
                    print_banner(f'LOADING WEIGHTS FROM STAGE {i}', '-', 19)
                    self.load_weights(f'{self.folder}/{self._name}_{i}.h5') 
                    continue

            if verbose:
                # Print staged model summary
                print_banner(f'STAGE {i}', '-')
                self.staged_models[i].summary()

                # Print out a message telling us which blocks are trainable
                for layer in self.staged_models[i].layers:
                    print(f'> {self.staged_models[i].name} layer {layer.name} is {"trainable" if layer.trainable else "not trainable"}')
                print('')

            # Compile staged model
            self.staged_models[i].compile(loss=self.loss, optimizer=self.optimizer)

            # Train staged model and save weights for this stage
            es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            hist = self.staged_models[i].fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es], verbose=verbose)
            self.save_weights(f'{self.folder}/{self._name}_{i}.h5')

            # Plot learning curve
            saveas = f'{self.folder}/{self._name}_{i}_history.png' if savefigs else None
            self.plot_learning_curve(hist, es, title=f'Autoencoder Learning Curve (Stage {i})', saveas=saveas)

            # Plot model outputs for batch of validation images
            saveas = f'{self.folder}/{self._name}_{i}_outputs.png' if savefigs else None
            self.plot_model_outputs(val_ds, layer=i, saveas=saveas)

        # Save weights for the full model after resetting all layers to be trainable
        self.trainable = True
        self.save_weights(f'{self.folder}/{self._name}.h5')

        # If we made it this far, we can safely delete the staged model checkpoints
        for i in range(self.encode_blocks):
            os.remove(f'{self.folder}/{self._name}_{i}.h5')


    def fit_end_to_end(self, 
                       train_ds, 
                       val_ds, 
                       epochs=150,
                       patience=10, 
                       savefigs=True,
                       verbose=1):
        '''
        Standard end-to-end training of autoencoder model. This is simply a
        wrapper around the Keras built-in fit method that also generates logs,
        checkpoints, and plots.
        '''
        # Create log directory for this model if it does not already exist
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Save model configuration in log directory
        self.save_config()

        # Helper routine for printing messages to the console during training
        print_banner = lambda msg, char, num=30: print('\n' + char*num + f' {msg} ' + char*num + '\n')
        if verbose: print_banner('BEGINNING END-TO-END TRAINING PROCESS', '=', 15)

        # Train model and save weights
        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        hist = self.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es], verbose=verbose)
        self.save_weights(f'{self.folder}/{self._name}_end_to_end.h5')

        # Plot learning curve
        saveas = f'{self.folder}/{self._name}_history_end_to_end.png' if savefigs else None
        self.plot_learning_curve(hist, es, title=f'Autoencoder Learning Curve (End-to-End)', saveas=saveas)

        # Plot model outputs for batch of validation images at each layer
        for i in range(self.encode_blocks):
            saveas = f'{self.folder}/{self._name}_{i}_outputs_end_to_end.png' if savefigs else None
            self.plot_model_outputs(val_ds, layer=i, saveas=saveas)


    def fine_tune(self, 
                  train_ds, 
                  val_ds, 
                  epochs=150,
                  patience=10, 
                  savefigs=True,
                  verbose=1):
        '''
        Fine-tunes the entire autoencoder model after greedy layer-wise training.
        '''
        # Helper routine for printing messages to the console during training
        print_banner = lambda msg, char, num=30: print('\n' + char*num + f' {msg} ' + char*num + '\n')
        if verbose: print_banner('BEGINNING FINE-TUNING PROCESS', '=', 19)

        # Load previous model weights from greedy training
        if verbose: print_banner(f'LOADING MODEL WEIGHTS', '-', 23)
        self.load_weights(f'{self.folder}/{self._name}.h5') 

        # Train model and save fine-tuned weights
        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        hist = self.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es], verbose=verbose)
        self.save_weights(f'{self.folder}/{self._name}_fine_tuned.h5')

        # Plot learning curve
        saveas = f'{self.folder}/{self._name}_history_fine_tuned.png' if savefigs else None
        self.plot_learning_curve(hist, es, title=f'Autoencoder Learning Curve (Fine Tuning)', saveas=saveas)

        # Plot model outputs for batch of validation images at each layer
        for i in range(self.encode_blocks):
            saveas = f'{self.folder}/{self._name}_{i}_outputs_fine_tuned.png' if savefigs else None
            self.plot_model_outputs(val_ds, layer=i, saveas=saveas)


    ##
    ## PREDICTION AND EVALUATION ROUTINES
    ##

    def predict_on_batch_at_layer(self, input_images, layer=-1):
        '''
        Returns latent codes and output images for a batch of input images at
        the specified layer.
        '''
        staged_encoder = Model(self.encoder.input, self.encoder.layers[layer].output)
        output_images = self.staged_models[layer].predict_on_batch(input_images)
        latent_codes = staged_encoder.predict_on_batch(input_images)

        return latent_codes, output_images


    def per_image_loss(self, input_images, output_images):
        '''
        Computes separate loss values for every image in a batch (i.e., without batch reduction).
        '''
        if self.loss.name == 'mean_squared_error':
            return np.mean((input_images - output_images)**2, axis=(1, 2, 3))

        elif self.loss.name == 'mean_absolute_error':
            return np.mean(np.abs(input_images - output_images), axis=(1, 2, 3)) 

        elif self.loss.name == 'ssim_loss':
            return tf.image.ssim(input_images, output_images, max_val=1.0)

        else:
            raise ValueError(f'unknown loss {self.loss.name}')   


    def evaluate_at_layer(self, ds, layer=-1):
        '''
        Computes loss for every image in the dataset at the specified layer.
        '''
        losses = []
        for input_images, _ in ds:
            output_images = self.staged_models[layer].predict_on_batch(input_images)
            losses.append(self.per_image_loss(input_images, output_images))

        return np.concatenate(losses)


    def get_latent_codes_at_layer(self, ds, layer=-1):
        '''
        Returns latent codes for every image in the dataset at the specified layer.
        '''
        latent_codes = []
        staged_encoder = Model(self.encoder.input, self.encoder.layers[layer].output)
        for input_images, _ in ds:
            latent_codes.append(staged_encoder.predict_on_batch(input_images))

        return np.concatenate(latent_codes)


    def latent_code_shape_at_layer(self, layer=-1):
        '''
        Returns the shape of the latent codes at the specified layer.
        '''
        return self.encoder.layers[layer].get_output_shape_at(0)[1:]


    def compute_ssim_at_layer(self, ds, layer=-1):
        '''
        Computes SSIM for every image in the dataset at the specified layer.
        '''
        ssim = []
        for input_images, _ in ds:
            output_images = self.staged_models[layer].predict_on_batch(input_images)
            ssim.append(tf.image.ssim(input_images, output_images, max_val=1.0))

        return np.concatenate(ssim)

    ##
    ## PLOTTING ROUTINES
    ##

    def plot_loss_at_layer(self, 
                           ds_dict,
                           layer=-1, 
                           saveas=None):
        '''
        Plots histograms of loss values for each dataset at the specified layer.
        '''
        # Convert negative index into correct positive index
        layer = list(range(self.encode_blocks))[layer]
        latent_dim = np.prod(self.latent_code_shape_at_layer(layer))

        # Evaluate loss on each dataset at this layer
        loss_dict = {label: self.evaluate_at_layer(ds, layer) for label, ds in ds_dict.items()}

        # Plot histogram of model loss breakdown
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, loss in loss_dict.items():
            ax.hist(loss, bins=25, alpha=0.5, density=True, label=label);
        xlabel_dict = {'mean_squared_error': 'Mean Squared Error', 
                       'mean_absolute_error': 'Mean Absolute Error', 
                       'ssim_loss': 'SSIM Loss'}
        ax.set_xlabel(xlabel_dict.get(self.loss.name, 'Loss'), fontsize=12);
        ax.set_ylabel('Density', fontsize=12);
        ax.set_title(f'Autoencoder Performance (Layer {layer}, Dim = {latent_dim:,})', fontsize=12)
        ax.legend();
        ax.grid(alpha=0.3);

        # Save figure if requested
        if saveas is not None:
            fig.savefig(saveas)

        return loss_dict


    def plot_loss(self,
                  ds_dict,
                  layers='all',
                  savefigs=True):
        '''
        Plots histograms of loss values for each dataset at *all* layers (or subset).
        '''
        if layers == 'all': layers = range(self.encode_blocks)

        loss_dicts = dict() 
        for layer in layers:
            saveas = f'{self.folder}/{self._name}_{layer}_loss.png' if savefigs else None
            loss_dicts[layer] = self.plot_loss_at_layer(ds_dict, layer=layer, saveas=saveas)

        return loss_dicts


    def plot_ssim_at_layer(self,
                           ds_dict,
                           layer=-1, 
                           saveas=None):
        '''
        Plots histograms for SSIM values for each dataset at the specified
        layer. Note: this is NOT the same as SSIMLoss! SSIMLoss = 1 - SSIM.
        '''
        # Convert negative index into correct positive index
        layer = list(range(self.encode_blocks))[layer]
        latent_dim = np.prod(self.latent_code_shape_at_layer(layer))

        # Compute SSIM on each dataset at this layer
        ssim_dict = {label: self.compute_ssim_at_layer(ds, layer) for label, ds in ds_dict.items()}

        # Plot histogram of model ssim breakdown
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, ssim in ssim_dict.items():
            ax.hist(ssim, bins=25, alpha=0.5, density=True, label=label);
        ax.set_xlabel('SSIM', fontsize=12);
        ax.set_ylabel('Density', fontsize=12);
        ax.set_title(f'Autoencoder Performance (Layer {layer}, Dim = {latent_dim:,})', fontsize=12)
        ax.legend();
        ax.grid(alpha=0.3);

        # Save figure if requested
        if saveas is not None:
            fig.savefig(saveas)

        return ssim_dict


    def plot_ssim(self,
                  ds_dict,
                  layers='all',
                  savefigs=True):
        '''
        Plots histograms for SSIM values for each dataset at *all* layers (or
        subset). Note: this is NOT the same as SSIMLoss! SSIMLoss = 1 - SSIM.
        '''
        if layers == 'all': layers = range(self.encode_blocks)

        ssim_dicts = dict()
        for layer in layers:
            saveas = f'{self.folder}/{self._name}_{layer}_ssim.png' if savefigs else None
            ssim_dicts[layer] = self.plot_ssim_at_layer(ds_dict, layer=layer, saveas=saveas)

        return ssim_dicts


    @staticmethod
    def plot_learning_curve(hist, early_stopping, title=None, saveas=None):
        '''
        Plots loss history for training and validation sets.
        '''
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(hist.history['loss'], label='Training')
        ax.plot(hist.history['val_loss'], label='Validation')
        if hasattr(early_stopping, 'best_epoch'):
            ax.axvline(x=early_stopping.best_epoch, linestyle='--', color='r', label=f'Best Model')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        if title is not None: 
            ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(alpha=0.4)

        # Save figure if requested
        if saveas is not None:
            fig.savefig(saveas)


    def plot_model_outputs(self, ds, layer=-1, saveas=None, nrows=8):
        '''
        Plots grid of `nrows` input images, reshaped latent codes, and output
        images side-by-side.
        '''
        # Attempt to grab the first `nrows` images from the dataset.
        try:
            input_images = next(ds.as_numpy_iterator())[0][:nrows]
        except:
            raise ValueError(f'nrows must be less than batch_size')

        # Extract latent codes and output images.
        latent_codes, output_images = self.predict_on_batch_at_layer(input_images, layer=layer)

        # Generate grid of appropriate size.
        fig, axes = plt.subplots(nrows, 3, figsize=(15, 5*nrows))
        for ax, input_image, latent_code, output_image in zip(axes, input_images, latent_codes, output_images):

            # Reshape latent code into square image with zero padding, if necessary
            latent_dim = np.prod(latent_code.shape)
            latent_image_size = np.ceil(np.sqrt(latent_dim)).astype(int)
            latent_image = np.zeros(latent_image_size**2)
            latent_image[:latent_dim] = latent_code.ravel()
            latent_image = latent_image.reshape(-1, latent_image_size)

            # Plot inputs, latent codes, and outputs side-by-side
            ax[0].imshow(input_image)
            ax[0].set_title('Input image')
            ax[1].imshow(latent_image)
            ax[1].set_title('Reshaped latent code')
            ax[2].imshow(output_image)
            ax[2].set_title('Output image')

            # Save figure if requested
            if saveas is not None:
                fig.savefig(saveas)


##
## CUSTOM AUTOENCODER LOSS FUNCTIONS
##

class SSIMLoss():
    '''
    Custom SSIM loss function for autoencoder
    '''
    def __init__(self):
        super().__init__()
        self.name = 'ssim_loss'

    def __call__(self, y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


##
## AUTOENCODER MODEL DEMO
##

if __name__ == '__main__':

    K.clear_session()
    demo = Autoencoder()
    demo.build((None, *IMAGE_SHAPE))
    demo.summary()
