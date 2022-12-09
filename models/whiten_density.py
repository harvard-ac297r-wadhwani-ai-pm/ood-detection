import random
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as BGM
import numpy as np

class whiten_density(): 
    '''
    Class to load latent codes, whiten, get density estimation
    '''
    def __init__(self, 
                 layer=4, 
                 whiten='yes', #yes: whiten, #no: no whitening 
                 p_white=0.8, #proportion of test data used to whiten
                 p_features = 0.7, #proportion of test data features used to derive bias and kernel 
                 density_estimator = 'BGM', #BGM: Bayesian Gaussian mixture, GMM: GaussianMixture
                 bgm_gmm_covariance = 'diag', #{‘full’, ‘tied’, ‘diag’, ‘spherical’}
                 bgm_gmm_n_components = 8, 
                 train_latent_dicts = dict(), 
                 test_latent_dicts = dict(), 
                 random_seed = 297):
      self.layer = layer 
      self.whiten = whiten
      self.p_white = p_white 
      self.p_features = p_features 
      self.density_estimator = density_estimator
      self.bgm_gmm_covariance = bgm_gmm_covariance
      self.bgm_gmm_n_components = bgm_gmm_n_components
      self.train_latent_dicts = train_latent_dicts
      self.test_latent_dicts = test_latent_dicts
      self.random_seed = random_seed

      assert self.whiten.lower() in ['yes', 'no'], 'whiten should be yes or no'
    
    def load_latent_codes(self):
      layer = self.layer 
      latent_codes_train_ID = self.train_latent_dicts[layer]['Train ID'].reshape(self.train_latent_dicts[layer]['Train ID'].shape[0],-1)
      latent_codes_train_OOD = self.train_latent_dicts[layer]['Train OOD'].reshape(self.train_latent_dicts[layer]['Train OOD'].shape[0],-1)
      latent_codes_test_ID = self.test_latent_dicts[layer]['Test ID'].reshape(self.test_latent_dicts[layer]['Test ID'].shape[0],-1)
      latent_codes_test_OOD = self.test_latent_dicts[layer]['Test OOD'].reshape(self.test_latent_dicts[layer]['Test OOD'].shape[0],-1)

      return latent_codes_train_ID, latent_codes_train_OOD, latent_codes_test_ID, latent_codes_test_OOD 

    def split_test_data_white(self):
      _, latent_codes_train_OOD, _, _ = self.load_latent_codes()
      
      np.random.seed(self.random_seed) 
      p = self.p_white 
      nrows = round(p*latent_codes_train_OOD.shape[0])
      idx = np.random.choice(latent_codes_train_OOD.shape[0], size=nrows, replace = False)
     
      latent_codes_train_OOD_white = np.take(latent_codes_train_OOD, idx, axis = 0)

      return latent_codes_train_OOD_white

    def compute_kernel_bias(self, vecs):
      out_dim= round(vecs.shape[-1]*self.p_features)
      mu = vecs.mean(axis=0, keepdims=True)
      cov = np.cov(vecs.T)
      u, s, vh = np.linalg.svd(cov)
      W = np.dot(u, np.diag(1 / np.sqrt(s)))
      return W[:, :out_dim], -mu

    def transform_and_normalize(self, vecs, bias, kernel):
      vecs = (vecs + bias).dot(kernel)
      return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def whitening(self): 
      latent_codes_train_ID, latent_codes_train_OOD, latent_codes_test_ID, latent_codes_test_OOD  = self.load_latent_codes()
      latent_codes_train_OOD_white = self.split_test_data_white()
      kernel, bias = self.compute_kernel_bias(latent_codes_train_OOD_white) #only using subset of test data for whitening
    
      if self.whiten.lower() == 'yes':
        feats_train_ID = self.transform_and_normalize(latent_codes_train_ID, bias, kernel)
        feats_train_OOD = self.transform_and_normalize(latent_codes_train_OOD, bias, kernel)
        feats_test_ID = self.transform_and_normalize(latent_codes_test_ID, bias, kernel)
        feats_test_OOD = self.transform_and_normalize(latent_codes_test_OOD, bias, kernel)
      
      elif self.whiten.lower() == 'no':
        feats_train_ID = latent_codes_train_ID
        feats_train_OOD = latent_codes_train_OOD
        feats_test_ID = latent_codes_test_ID
        feats_test_OOD = latent_codes_test_OOD

      return feats_train_ID, feats_train_OOD, feats_test_ID, feats_test_OOD

    def density_est(self): 
      #estimator 
      if self.density_estimator.lower() == 'bgm':  
        estimator = BGM(weight_concentration_prior_type="dirichlet_process", covariance_type = self.bgm_gmm_covariance, 
                n_components=self.bgm_gmm_n_components, reg_covar=1e-6, init_params='kmeans',
                max_iter=1000, mean_precision_prior=.5, random_state=2)
      elif self.density_estimator.lower() == 'gmm': 
        estimator = GMM(n_components=self.bgm_gmm_n_components, init_params='kmeans', covariance_type=self.bgm_gmm_covariance, random_state=2)  
      
      #fit estimator
      feats_train_ID, feats_train_OOD, feats_test_ID, feats_test_OOD = self.whitening()
      estimator.fit(feats_train_ID)

      #calculate loglikelihood 
      scores_train_ID = -estimator.score_samples(feats_train_ID)
      scores_train_OOD = -estimator.score_samples(feats_train_OOD)
      scores_test_ID = -estimator.score_samples(feats_test_ID)
      scores_test_OOD = -estimator.score_samples(feats_test_OOD)

      return scores_train_ID, scores_train_OOD, scores_test_ID, scores_test_OOD