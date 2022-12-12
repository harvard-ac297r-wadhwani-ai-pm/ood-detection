import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, f1_score


def distribution_overlap_summary(input_dict:dict,bins:np.array,plot_heatmap=False,save_fig_path=None,ax=None)->pd.DataFrame:
    """
    Computes the distribution overlap summary for a set of distributions provided. The distribution overlap summary
    is a k by k table for an input of k distributions which describes how much each overlaps with the others in a
    pairwise comparison. The main diagonal entries describe what % of the histogram area for the ith distribution
    is uniquely its own. The entries of each row describe what % of the distributions histogram is shared with each
    of the other distributions. Each value in the table is bounded between 0 and 1, but each row could potentially 
    sum to more than once since overlap between distributions can occur for more than 2 distributions at once

    Parameters
    ----------
    input_dict : dict
        A dictionary containing the distribution labels and keys and the observations from each distribution as values.
    bins : np.array
        A numpy array describing the end points of the bins, this will be used to compute N-1 bin counts for each distribution.

    Returns
    -------
    output_array : pd.DataFrame
        Returns a k by k summary table as described above, on row and column for each distribution in the input dict.
        Each row describes what % of that distribution's histogram area is shared with other distributions i.e. what 
        % overlap there is
    """
    k = len(input_dict) # Count the number of distributions being used
    pct_dist_by_bin_arr = np.zeros([len(bins)-1,k]) # Create a N-1 x K array for N bins and K distributions
    
    for j, values in enumerate(input_dict.values()):
      counts, edges = np.histogram(values, bins) # Compute the number of obs falling into each bin
      pct_dist_by_bin_arr[:,j] = counts/counts.sum() # Store what % of the obs fall into each bin by distribution
    
    output_array = np.zeros([k,k]) # Create an output array that is kxk where k = # of distributions
    
    for j in range(k):
      diffs_array = np.subtract(pct_dist_by_bin_arr, pct_dist_by_bin_arr[:,j].reshape(-1,1)) # Compute each distribution col minus the jth distribution col
      # This gives us how much more of the jth class is in each bin on a % basis
      diffs_array[diffs_array<=0] = 0 # Erase negative values so that we can count the overlap, negative values indicate the jth distribution has a lesser
      # amount of a % of its observations falling into this bin which should be recorded as 100% overlap, here we are calculating how much of the jth
      # distribution is exceeding each of the others for each bin, we will compute the overlap as (1 - area exceeding)
      non_overlap = np.delete(diffs_array,j,axis=1).sum(axis=0) # Take the column sum - these are the values for the elements other than j
      # Compute the overlap as 1 minus the non-overlaping amount in each bin vs each of the other distributions
      output_array[j,[i!=j for i in range(k)]] = 1 - non_overlap
    
      pct_dist_by_bin_arr_copy = pct_dist_by_bin_arr.copy() # Make a copy of the distribution array
      pct_dist_by_bin_arr_copy[:,j]=0 # Zero out the jth column
      max_vals = pct_dist_by_bin_arr_copy.max(axis=1) # Compute the max for each row representing the max potential overlap for each bin
      diffs_vec = pct_dist_by_bin_arr[:,j] - max_vals # Compute the difference in values for each bin giving us how much this dist is in excess of all others
      diffs_vec[diffs_vec<=0] = 0 # Zero out any negative values
      output_array[j,j] = diffs_vec.sum() # Compute the area of the distribution that is uniquely this distribution only
     
    # Convert to a pandas dataframe, add labels and return the the user
    output_array = pd.DataFrame(output_array);output_array.index=list(input_dict.keys());output_array.columns=list(input_dict.keys())
    if plot_heatmap:
      if ax is None:
        fig,ax = plt.subplots()
      sn.heatmap(output_array,cmap="Reds",annot=output_array.round(2),cbar=True, fmt = '',ax=ax)
      ax.set_title("Distribution Overlap Summary")
      if save_fig_path is not None:
        fig.savefig(save_fig_path)

    return output_array



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
        
        
def plot_SSIM_summary(ID_ssim_array=None, EC_ssim_array=None, OOD_ssim_array=None,compute_max=True):
  fig,axes = plt.subplots(1,3,figsize=(20,4))

  if ID_ssim_array is not None:
    if compute_max:
      ID_ssim_max = ID_ssim_array.mean(axis=1) # Compute the mean SSIM score vs the in-sample comparison batch for each image
      axes[2].scatter(ID_ssim_max,ID_ssim_array.std(axis=1),label="ID",alpha=0.25,s=10) # Make a scatter plot for the mean SSID value vs the ID comp set vs the sd of the SSID values
    else:
      ID_ssim_max = ID_ssim_array
    axes[0].hist(ID_ssim_max,label="ID",alpha=0.5,density=True,bins=35) # Overlapping histogram chart showing mean SSID vs ID comparison set
    sn.kdeplot(ID_ssim_max,label="ID",ax=axes[1]) # Plot as a kernel density estimator plot

  if EC_ssim_array is not None:
    if compute_max:
      EC_ssim_max = EC_ssim_array.mean(axis=1) # Compute the mean SSIM score vs the in-sample comparison batch for each image
      axes[2].scatter(EC_ssim_max,EC_ssim_array.std(axis=1),label="EC",alpha=0.25,s=10) # Make a scatter plot for the mean SSID value vs the ID comp set vs the sd of the SSID values
    else:
      EC_ssim_max = EC_ssim_array
    axes[0].hist(EC_ssim_max,label="EC",alpha=0.5,density=True,bins=35) # Overlapping histogram chart showing mean SSID vs ID comparison set
    sn.kdeplot(EC_ssim_max,label="EC",ax=axes[1]) # Plot as a kernel density estimator plot

  if OOD_ssim_array is not None:
    if compute_max:
      OOD_ssim_max = OOD_ssim_array.mean(axis=1) # Compute the mean SSIM score vs the in-sample comparison batch for each image
      axes[2].scatter(OOD_ssim_max,OOD_ssim_array.std(axis=1),label="OOD",alpha=0.25,s=10) # Make a scatter plot for the mean SSID value vs the ID comp set vs the sd of the SSID values
    else:
      OOD_ssim_max = OOD_ssim_array
    axes[0].hist(OOD_ssim_max,label="OOD",alpha=0.5,density=True,bins=35) # Overlapping histogram chart showing mean SSID vs ID comparison set
    sn.kdeplot(OOD_ssim_max,label="OOD",ax=axes[1]) # Plot as a kernel density estimator plot 
    

  # Overlapping histogram chart showing Max SSID vs ID comparison set
  axes[0].set_title("SSIM Max vs ID Comparison Set")
  axes[0].set_xlabel("SSIM Max Score");axes[0].set_ylabel("Density");axes[0].legend()

  # Plot as a kernel density estimator plot instead which is a bit easier to see / understand than the overlapping histograms alone
  axes[1].set_title("SSIM Max vs ID Comparison Set")
  axes[1].set_xlabel("SSIM Max Score");axes[1].set_ylabel("Density");axes[1].legend()

  if compute_max:
    # Make a scatter plot for the Max SSID value vs the ID comp set vs the sd of the SSID values across each of the comp set images
    axes[2].set_title("SSIM Max and Std Dev vs ID Comparison Set")
    axes[2].set_xlabel("Max SSIM vs ID Comp Set");axes[2].set_ylabel("Std Dev SSIM vs ID Comp Set");axes[2].legend()

  plt.tight_layout();plt.show()
  
  
def logistic_regression_summary(ID_ssim_mean, EC_ssim_mean, OOD_ssim_mean, save_model=False, show_dist_overlap_table=True, return_model=False):
  if show_dist_overlap_table:
    fig,axes = plt.subplots(1,3,figsize=(20,4.5))
  else:
    fig,axes = plt.subplots(1,2,figsize=(13,4.5))
  print("Logistic Regression Summary on SSIM Mean Values")

  # 1 Feature Logistic classifier using only the mean SSID score
  SSID_data = np.concatenate([ID_ssim_mean,OOD_ssim_mean]) # Combine the input feature data into 1 large np.array
  ID_OOD_labels = np.concatenate([np.ones(len(ID_ssim_mean)),np.zeros(len(OOD_ssim_mean))]) # Create a response vector of 0s and 1s for ID and EC respectively

  # Apply a train-test split to the data before applying a logistic regression model
  train_x, test_x, train_y, test_y = train_test_split(SSID_data, ID_OOD_labels, train_size=0.8, shuffle=True, random_state=297)
  train_x = pd.DataFrame(train_x);test_x = pd.DataFrame(test_x)

  # Train the 1 feature logistic regression modelmodel
  SSID_classifier = LogisticRegression(penalty="none",class_weight='balanced') # Weighted logistic regression to account for class imbalance
  SSID_classifier.fit(train_x,train_y) # Fit the logistic classifier on the training data set

  # Make predictions and compute the confusion matrix
  y_pred_train = SSID_classifier.predict(train_x).astype(int)
  y_pred_test = SSID_classifier.predict(test_x).astype(int)

  plot_confusion_matrix(train_y, y_pred_train, title='In-Sample Confusion matrix', label_text={0:"ID",1:"OOD"}, figsize=(5,5), color_by_row=True, saveas=None, ax=axes[0])
  plot_confusion_matrix(test_y, y_pred_test, title='Validation Confusion matrix', label_text={0:"ID",1:"OOD"}, figsize=(5,5), color_by_row=True, saveas=None, ax=axes[1])

  if save_model!=False:
    with open(save_model+'.pkl','wb') as f:
      pickle.dump(SSID_classifier,f)
  
  if show_dist_overlap_table==True:
    # Make a heatmap plot of the distribution overlap summary
    if EC_ssim_mean is not None:
      summary = distribution_overlap_summary({"ID":ID_ssim_mean,"EC":EC_ssim_mean,"OOD":OOD_ssim_mean},np.linspace(0.0,1.0,num=100),plot_heatmap=True,save_fig_path=None,ax=axes[2])
    else:
      summary = distribution_overlap_summary({"ID":ID_ssim_mean,"OOD":OOD_ssim_mean},np.linspace(0.0,1.0,num=100),plot_heatmap=True,save_fig_path=None,ax=axes[2])

  if return_model==True:
    return SSID_classifier

