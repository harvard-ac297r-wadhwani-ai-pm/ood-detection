import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
import seaborn as sn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse, imutils, cv2



def plot_learning_curve(hist, es, title='Mean Squared Error', saveas=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(hist.history['loss'], label='Training')
    ax.plot(hist.history['val_loss'], label='Validation')
    if hasattr(es, 'best_epoch'):
        ax.axvline(x=es.best_epoch, linestyle='--', color='r', label=f'Best Model')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(alpha=0.4)

    # Save figure if requested
    if saveas is not None:
        fig.savefig(saveas)
  
def wt_BCE_loss_debug_test(y_true, y_pred):
  """Custom defined weighted binary cross entropy loss function, relies on w_1 and w_0 defined weights on the 1s and 0s defined in the global frame"""
  y_true = tf.cast(y_true, tf.float32) # Need to cast y_true int32 type to float so that these can be multiplied together
  y_pred = tf.keras.backend.clip(y_pred, min_value=1e-6, max_value = 1 - 1e-6) # Use clipping to avoid 0s and 1s which will cause errors in tf.math.log()
  bce_loss_values_1 = y_true*tf.math.log(y_pred) # Compute binary cross entropy terms for the y_true==1 terms
  bce_loss_values_0 = (1-y_true)*tf.math.log(1-y_pred) # Compute binary cross entropy terms for the y_true==0 terms
  # Take the dot product with the w_1 weighted loss weights for the y_true==1 terms
  # Take the dot product with the w_0 weighted loss weights for the y_true==0 terms
  loss_total = bce_loss_values_1*w_1 + bce_loss_values_0*w_0 # Average loss by class
  return_val = -tf.reduce_mean(loss_total)
  if tf.math.is_nan(return_val):
    a=tf.print(y_true)
    a=tf.print(y_pred)
    a=tf.print(bce_loss_values_1)
    a=tf.print(bce_loss_values_0)
    a=tf.print(loss_total)
  return return_val

def wt_BCE_loss(y_true, y_pred):
  """Custom defined weighted binary cross entropy loss function, relies on w_1 and w_0 defined weights on the 1s and 0s defined in the global frame"""
  y_true = tf.cast(y_true, tf.float32) # Need to cast y_true int32 type to float so that these can be multiplied together
  y_pred = tf.keras.backend.clip(y_pred, min_value=1e-6, max_value = 1 - 1e-6) # Use clipping to avoid 0s and 1s which will cause errors in tf.math.log()
  bce_loss_values_1 = y_true*tf.math.log(y_pred) # Compute binary cross entropy terms for the y_true==1 terms
  bce_loss_values_0 = (1-y_true)*tf.math.log(1-y_pred) # Compute binary cross entropy terms for the y_true==0 terms
  # Take the dot product with the w_1 weighted loss weights for the y_true==1 terms
  # Take the dot product with the w_0 weighted loss weights for the y_true==0 terms
  loss_total = bce_loss_values_1*w_1 + bce_loss_values_0*w_0 # Average loss by class
  return -tf.reduce_mean(loss_total)


def predict_on_test(model, test_set_gen, write_to_csv=False, filename=None):
  """A function for computing the predictions of a model on the test set"""
  print("Generating test set predictions...") # Screen update
  test_set_predictions = [] # A list to hold the predictions made on the test set
  test_set_y_labels = [] # A list to hold the y_true labels of the test set
  for i in tqdm(range(len(test_set_gen))): # Loop over all batches in the test set 
    input_x_data, y_labels_data = next(test_set_gen) # Unpack into x and y
    test_set_predictions.append(model(input_x_data)) # Make predictions on the x input data of the batch
    test_set_y_labels.append(y_labels_data) # Append true y labels into a list

  # Concatenate into 1 large output tensor for both the model predictions and y_true labels 
  test_set_preds = tf.concat(test_set_predictions, 0) # Concatenate all batch predictions into 1 tensor
  test_set_y_true = tf.concat(test_set_y_labels, 0) # Concatenate all batch y_true labels into 1 tensor
  test_set_preds = test_set_preds[:,0] # Select the first column which is the only column
  assert test_set_y_true.shape == test_set_preds.shape, "y_pred and y_true tensor shape mismatch"

  if write_to_csv == True: # If write_to_csv is set to true, then save results to CSV
    combined_df = pd.DataFrame({"y_true":test_set_y_true.numpy(),"y_pred":test_set_preds.numpy()})
    assert filename is not None, "If write_to_csv is set to true, filename cannot be none"
    combined_df.to_csv(filename)

  test_set_preds = pd.DataFrame(test_set_preds.numpy()) # Convert to pd dataframe
  test_set_y_true = pd.DataFrame(test_set_y_true.numpy()) # Convert to pd dataframecon

  return test_set_preds, test_set_y_true # Outputs 2 Pandas DataFrames of size [Observations x Classes]


# A custom function for plotting confusion matricies
def plot_confusion_matrix(y_label, y_pred, title='Confusion matrix', label_text={0:"Class A",1:"Class B",2:"Class C"}, figsize=(10,10), color_by_row=True, saveas=None, ax=None):
    """Creates a confusion matrix as a heatmap using the y_labels and y_pred inputs"""
    unique_labels = np.unique(y_label) # Get the unique levels of the true labels
    unique_pred = np.unique(y_pred) # Get the unique levels of the prediced labels
    unique_labels = np.unique(np.hstack([unique_labels, unique_pred])) # Find the combined number of unique labels for both
    unique_labels = np.sort(unique_labels) # Sort unique labels in ascending order
    d = len(unique_labels) # Get the number of unique labels
    row_vectors = [] # Create a blank list to hold the row vectors
    for label in unique_labels:
      # Loop over each of the unique labels and compute the coressponding confusion matrix row
      row_filter = y_label==label # Get a T/F filter for where the true y is "label"
      row_vector = np.zeros(d) # Create a new row vector to append to the confusion matrix
      if np.sum(row_filter)>0:
        # If there is at least 1 such instance of this true y_label, compute what we predicted for it
        predictions = y_pred[row_filter] # Get the predictions corresponding to the y_true = label
        class_labels, counts = np.unique(predictions,return_counts=True) # Count the number of entries for each unique instance
        for idx,count in zip(class_labels,counts):
          # For each class label predicted, store the counts associated with it
          row_vector[idx] = count
        row_vectors.append(row_vector) # Append this row vector to the list of row vectors making up the matrix

    confusion_matrix_array = np.vstack(row_vectors) # Turn row vector list into a 2d array

    confusion_matrix_array = np.vstack(row_vectors) # Turn row vector list into a 2d array
    x_labels = [label_text[val] for val in unique_labels] # Create x-axis tick mark labels
    y_labels = [label_text[val] for val in unique_labels] # Create y-axis tick mark labels
    
    # Create annotations for each square of the grid to display the coutn and the % normalized by row
    row_percents = np.vstack([vector/np.sum(vector) for vector in row_vectors])
    annotations = [str(int(value))+"\n"+str(round(precent*100,2))+"%" for precent, value in zip(row_percents.ravel(), confusion_matrix_array.ravel())]
    annotations = np.array(annotations).reshape(d,d)
    
    # Create the heatmap plot using seaborn
    if ax is None:
      fig,ax = plt.subplots(1,1,figsize=(figsize))
    
    ax.set_title(title)
    if color_by_row==True:
      plot = sn.heatmap(row_percents,cmap="Blues",annot=annotations,cbar=True,xticklabels=x_labels, yticklabels=y_labels, fmt = '',ax=ax)
    else:
      plot = sn.heatmap(confusion_matrix_array,cmap="Blues",annot=annotations,cbar=True,xticklabels=x_labels, yticklabels=y_labels, fmt = '',ax=ax)
    plot.set(xlabel='Predicted Labels', ylabel='True Label');

    if saveas is not None:
        plt.savefig(saveas, dpi=300)

def generate_roc_summary(y_true, y_pred, ax=None):
  # Plot the ROC curve and display the AUC score in the title of the plot for quick reference
  fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

  if ax is None: # Check if a plotting axis has been provided
    fig,ax = plt.subplots(1,1,figsize=(8,6))
  plt.plot([0,1],[0,1], linestyle="--",label="Baseline")
  ax.plot(fpr, tpr,zorder=3,label="ROC curve")
  ax.set_xlabel("FPR");ax.set_ylabel("TPR");ax.grid(color='lightgray',zorder=-3)
  ax.legend();ax.set_title("ROC Curve - AUC Score = "+str(round(roc_auc_score(y_true, y_pred),3)))

  if ax is None:
    plt.show()


# Import needed functionalities from tf_keras_vis for saliency map visualization
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore

def predict_on_test_with_filenames(model, data_gen, num_batches):
  """A function for computing the predictions of a model for the batches of a data generator and saving those with filenames"""
  print("Generating model set predictions...") # Screen update
  # Set up aggregation data structures to track results as we batch and compute predictions form the model
  model_predictions = [] # A list to hold the predictions made by the model on batches from data_gen
  y_labels = [] # A list to hold the y_true labels of the batches from data_gen
  filename_list = [] # A list of the filenames for the input images we've recieved in batches from data_gen
  batch_imgs = [] # A list of input images from the batches

  all_filenames = np.array(data_gen.filenames) # Get a listing of all the filenames in data_gen
  for i in tqdm(range(num_batches)): # Loop over the number of batches specified by the user
    index = next(data_gen.index_generator) # Get the next batch of indices from the data generator
    input_x_data, y_labels_data = data_gen._get_batches_of_transformed_samples(index) # Get the batch images and image labels from the data_gen
    model_predictions.append(model(input_x_data)) # Make predictions on the x input data of the batch
    y_labels.append(y_labels_data) # Append true y labels into a list
    filename_list.append(all_filenames[index]) # Append the fileames that we iterated through
    batch_imgs.append(input_x_data) # Append the input images used in this process

  # Concatenate into 1 large output tensor for both the model predictions and y_true labels 
  model_predictions = tf.concat(model_predictions, 0) # Concatenate all batch predictions into 1 tensor
  y_labels = tf.concat(y_labels, 0) # Concatenate all batch y_true labels into 1 tensor 
  filename_list = np.concatenate(filename_list, 0) # Concatenate all the filenames of all the images used into 1 tensor
  batch_imgs = np.concatenate(batch_imgs,0) # Concatenate all the input images of all images used into 1 tensor
  assert y_labels.shape[0] == model_predictions.shape[0], "y_pred and y_true tensor shape mismatch"

  return model_predictions, y_labels, filename_list, batch_imgs # Outputs the values computed to the user


class SaliencyMapGen():
  # Documentation: https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html
  def __init__(self, model_name):
    replace2linear = ReplaceToLinear() # Create a saliency computation model
    self.saliency = Saliency(model_name, model_modifier=replace2linear, clone=True)
  def plot_saliency_map(self, input_image, y_pred, colormap = cv2.COLORMAP_HOT, alpha=0.5, saliency_only=False, ax=None, smooth_grad=True, smooth_samples=20,smooth_noise=20):
    """
    input_image: [l x w x 3] np.array of the input image pixel values, values must be within [0,1]
    y_pred: The prediction from the model, should be an int {0,1}
    colormap: The colormap for the saliency map, is used to convert a [l x w] heatmap of [0,1] values to [l x w x 3]
    alpha: The blending factor of the original image and the new image
    saliency_only: Boolean parameter controlling if the function should plot just the saliency map without the original image
    ax: The plot axis to place the saliency plot on, if None provided, a new plot space will be created
    smooth_grad: Boolean parameter controlling the usage of smooth grad
    smooth_samples / smooth_noise: Smooth grad smoothing parameters
    """
    if smooth_grad==True:
      saliency_map = self.saliency(BinaryScore(y_pred) , input_image, smooth_samples=smooth_samples, smooth_noise=smooth_noise) # Smooth grad saliency map
    else:
      saliency_map = self.saliency(BinaryScore(y_pred) , input_image) # Without smooth grad
    
    saliency_map = saliency_map[0,:,:] # Remove the leading dimension referring to # images, reduce down to [l x w]
    saliency_map_overlay = (cv2.applyColorMap((saliency_map*256).astype(np.uint8), colormap)).astype(np.float32)/256 # Convert from [l x w] into [l x w x 3] and colorize the heatmap
    combined_output = cv2.addWeighted(input_image, alpha, saliency_map_overlay, 1 - alpha, 0) # Combine the original image and the saliency map to make a new visual

    if ax is None:
      fig,ax = plt.subplots(1,1)
    
    if saliency_only:
      ax.imshow(saliency_map_overlay)
    else:
      ax.imshow(combined_output)

# Code adapted from: https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
# Original Paper: https://arxiv.org/abs/1610.02391

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = (cv2.applyColorMap(heatmap, colormap)/256).astype(np.float32)
        if len(image.shape)==4:
          image = image[0,:,:,:]
        elif len(image.shape)!=3:
          raise TypeError("Image dim not equal to 3")
        output = cv2.addWeighted(image.astype(np.float32), alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)