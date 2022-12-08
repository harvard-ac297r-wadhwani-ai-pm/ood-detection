from tqdm import tqdm
import cv2
import skimage.measure 
from skimage.measure import shannon_entropy
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import det_curve, DetCurveDisplay, f1_score, confusion_matrix, accuracy_score,roc_auc_score,roc_curve, classification_report, RocCurveDisplay
from sklearn.metrics import precision_recall_fscore_support as score
from imblearn.over_sampling import SMOTE 
from collections import Counter
from xgboost import XGBClassifier

def simple_model_generate_features(df,ood=False,pct_whiteish_threshold=0.9):
  '''
  Input: images
  Output: Feature vector for mean, correlation, entropy, contrast, and pct_whiteish pixels 
  '''
  #instantiate empty lists for features 
  r_avg_list = []
  g_avg_list = []
  b_avg_list = []

  r_var_list = []
  g_var_list = []
  b_var_list = []

  h_avg_list = []
  s_avg_list = []
  l_avg_list = []

  h_var_list = []
  s_var_list = []
  l_var_list = []

  r_g_corr_list = []
  r_b_corr_list = []
  g_b_corr_list = []
  entropy_list = []
  contrast_list = []
  pct_whiteish_list = []

  for batch_ndx, sample in enumerate(tqdm(df)):
    try:
      #fetch a single batch of 32
      batch_imgs = sample[0]
      for i in range(32):
        #extracts each color channel 
        r_init = batch_imgs[i][:,:,0].numpy().flatten()
        g_init = batch_imgs[i][:,:,1].numpy().flatten()
        b_init = batch_imgs[i][:,:,2].numpy().flatten()

        #calculate the mean RGB values
        r_avg = np.mean(r_init)
        g_avg = np.mean(g_init)
        b_avg = np.mean(b_init)

        #calculate the variance RGB values
        r_var = np.var(r_init)
        g_var = np.var(g_init)
        b_var = np.var(b_init)

        #calculates the correlation between R-G, R-B, and G-B 
        r_g_corr = np.corrcoef(r_init,g_init,rowvar=False)[0,1]
        r_b_corr = np.corrcoef(r_init,b_init,rowvar=False)[0,1]
        g_b_corr = np.corrcoef(g_init,b_init,rowvar=False)[0,1]


        #calculates the entropy for the individual image int he batch 
        entropy = shannon_entropy(batch_imgs[i])

        #calculates RMS contrast - https://en.wikipedia.org/wiki/Contrast_(vision)#RMS_contrast
        img_grey = cv2.cvtColor(np.array(batch_imgs[i]), cv2.COLOR_BGR2GRAY)
        contrast = img_grey.std()


        #checks to confirm each pixel across all three channels is above threshold 
        r_mask = r_init > pct_whiteish_threshold 
        g_mask = g_init > pct_whiteish_threshold 
        b_mask = b_init > pct_whiteish_threshold 
        all_whiteish = r_mask & g_mask & g_mask
        pct_whiteish = np.sum(all_whiteish) / len(all_whiteish)

         # Convert the ID, EC and OOD images sets from RGB to HLS
        img_hsl = cv2.cvtColor(np.array(batch_imgs[i]), cv2.COLOR_RGB2HLS)
  
        #extracts each color channel 
        h_init = img_hsl[:,:,0].flatten()
        s_init = img_hsl[:,:,1].flatten()
        l_init = img_hsl[:,:,2].flatten()
  

        

        #calculate the mean RGB values
        h_avg = np.mean(h_init)
        s_avg = np.mean(s_init)
        l_avg = np.mean(l_init)

        #calculate the variance RGB values
        h_var = np.var(h_init)
        s_var = np.var(s_init)
        l_var = np.var(l_init)

       
        #appends all to list
        r_avg_list.append(r_avg)
        g_avg_list.append(g_avg)
        b_avg_list.append(b_avg)

        r_var_list.append(r_var)
        g_var_list.append(g_var)
        b_var_list.append(b_var)

        r_g_corr_list.append(r_g_corr)
        r_b_corr_list.append(r_b_corr)
        g_b_corr_list.append(g_b_corr)

        entropy_list.append(entropy)
        contrast_list.append(contrast)
        pct_whiteish_list.append(pct_whiteish)

        h_avg_list.append(h_avg)
        s_avg_list.append(s_avg)
        l_avg_list.append(l_avg)

        h_var_list.append(h_var)
        s_var_list.append(s_var)
        l_var_list.append(l_var)
        
    except:
      pass

  df_idx = np.arange(len(r_avg_list))
  df_x = np.vstack((r_avg_list,
                    g_avg_list,
                    b_avg_list,
                    r_var_list,
                    g_var_list,
                    b_var_list,
                    r_g_corr_list,
                    r_b_corr_list,
                    g_b_corr_list,
                    entropy_list,
                    contrast_list,
                    pct_whiteish_list,
                    h_avg_list,
                    s_avg_list,
                    l_avg_list,
                    h_var_list,
                    s_var_list,
                    l_var_list,                    
                    )).T
  # if OOD, then zero; if ID then 1
  if ood is True: 
    df_y = np.zeros(df_x.shape[0])
  else:
    df_y = np.ones(df_x.shape[0])
  
  return df_x, df_y


#generate features
def load_split_scale(df_train_ds, df_test_ds,pct_whiteish_threshold=0.9):
  '''
  objective: load data, split into test/train, and scale
  input: train data loader, test data loader, threshold 
  returns: X_train, X_test, y_train, y_test 
  '''
  t = pct_whiteish_threshold
  print("Generating features for ood")
  ood_x, ood_y = simple_model_generate_features(df_test_ds,ood=True,pct_whiteish_threshold=t)

  print("Generating features for id")
  train_x, train_y = simple_model_generate_features(df_train_ds,ood=False,pct_whiteish_threshold=t)
  
  #concats id and ood
  total_train = np.vstack((train_x,ood_x))
  total_y = np.vstack((train_y.reshape(-1,1),ood_y.reshape(-1,1)))
  
  #test train split and scales data to have mean 0 
  X_train, X_test, y_train, y_test = train_test_split(total_train, total_y, test_size=0.33, random_state=42)
  
  #instantiates and fit standard scaler 
  scaler = StandardScaler().fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  #dealing with dropping nulls that occur -- very rare 
  X_train_clean = X_train_scaled[~np.isnan(X_train_scaled).any(axis=1)]
  y_train_clean = y_train[~np.isnan(X_train_scaled).any(axis=1)]
  X_test_clean = X_test_scaled[~np.isnan(X_test_scaled).any(axis=1)]
  y_test_clean = y_test[~np.isnan(X_test_scaled).any(axis=1)]
  
  return X_train_clean, X_test_clean, y_train_clean, y_test_clean, scaler


def plot_roc_curve(y_true, y_pred_probs, name='simple model'):
  auc_score = round(roc_auc_score(y_true, y_pred_probs),3)
  fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
  ns_probs = [0 for _ in range(len(y_true))]
  fpr_ns, tpr_ns, thresholds_ns = roc_curve(y_true, ns_probs)
  fig, ax = plt.subplots(1, 1)
  ax.plot(fpr, tpr,label=name)
  ax.plot(fpr_ns,tpr_ns,label='no skill')
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title(f'ROC curve for test set: \n AUC {auc_score}')
  ax.legend()
  plt.grid(alpha=0.3)
  plt.show()

def plot_det_curve(y_true, y_pred_probs,name='simple model'):
  fpr, tpr, thresholds = det_curve(y_true, y_pred_probs)
  fig, ax = plt.subplots(1, 1)
  ax.plot(fpr, tpr,label=name)
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('False Negative Rate')
  ax.set_title(f'DET curve for test set')
  ax.set_xlim([0, 0.25])
  ax.set_ylim([0, 0.25])
  plt.grid(alpha=0.3)
  ax.legend()
  plt.show()

def xgbmodel_metrics(X_train, X_test, y_train, y_test):
  #instantiate and fit xgboost model
  clf = XGBClassifier(learning_rate=0.02, 
                      n_estimators=600, 
                      objective='binary:logistic',
                    silent=True, 
                    nthread=1,
                    booster='gbtree',random_state=0)
  
  print("XGBoost + SMOTE:")
  sm = SMOTE(random_state=42)
  X_res, y_res = sm.fit_resample(X_train, y_train)
  clf.fit(X_res, y_res.ravel())
  #compute metrics 
  return_metrics(clf, X_test, y_test,name='XGBoost')
  return clf

def return_metrics(clf, X_test, y_test, name='model'):
  #return test metrics  
  y_test_preds = clf.predict(X_test)
  
  cf_matrix = confusion_matrix(y_test, y_test_preds)
  target_names = ['out-of-distribution','in-distribution']
  print("AUC:",roc_auc_score(y_test.ravel(), clf.predict_proba(X_test)[:, 1]))
  print("F1:",f1_score(y_test, y_test_preds, average='macro'))
  diag = cf_matrix.diagonal()/cf_matrix.sum(axis=1)
  print("Accuracy ID:",diag[1])
  print("Accuracy OOD:",diag[0])
  
  precision,recall,fscore,support=score(y_test, y_test_preds)
  print("Classification report:\n",classification_report(y_test, y_test_preds,target_names=target_names))
  print("Accuracy:",accuracy_score(y_test, y_test_preds))
  print("Precision OOD:",precision[1])
  print("Recall OOD:",recall[1])
  
  #print confusion matrix seaborn 
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ["{0:0.0f}".format(value) for value in
                  cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds',xticklabels=['OOD','ID'], yticklabels=['OOD','ID']);

  plot_roc_curve(y_test, clf.predict_proba(X_test)[:, 1],name)
  plot_det_curve(y_test, clf.predict_proba(X_test)[:, 1],name)

  #print coefficients 
  try:
    coef_df = pd.DataFrame()
    coef_df['features'] = ['r_avg','g_avg','b_avg','r_var','g_var','b_var','r_g_corr', 'r_b_corr','g_b_corr','entropy','contrast','pct_whiteish',
                           'h_avg','s_avg','l_avg','h_var','s_var','l_var',
                           ]
    coef_df['weights'] = clf.coef_.ravel()
    coef_df['weights_abs'] = np.abs(coef_df['weights'])
    coef_df = coef_df.sort_values(by='weights_abs',ascending=False)
    display(coef_df)
  except:
    pass

def logistic_regression_model_metrics(X_train, X_test, y_train, y_test,smote_flag=False):
  if smote_flag:
    print("Logistic Regression + SMOTE:")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    clf = LogisticRegression(random_state=0).fit(X_res, y_res)
    return_metrics(clf, X_test, y_test, name = 'SMOTE model')
  else:
    print("Logitic Regression with no SMOTE:")
    clf = LogisticRegression(random_state=0).fit(X_train, y_train.ravel())
    return_metrics(clf, X_test, y_test, name = 'No SMOTE model')

  return clf