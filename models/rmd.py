from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as BGM
import numpy as np

class relative_mahalanobis_distance(): 
    '''
    Class to load latent codes, calculate Relative Mahalanobis Distance (RMD)
    '''
    def __init__(self, 
                 layer = 4, 
                 density_estimator = 'BGM', #BGM: Bayesian Gaussian mixture, GMM: GaussianMixture 
                 bgm_gmm_covariance = 'diag', #{‘tied’, ‘diag’, ‘spherical’}
                 bgm_gmm_n_components = 8, 
                 train_latent_dicts = dict(), 
                 test_latent_dicts = dict(), 
                 random_seed = 297):
      self.layer = layer 
      self.density_estimator = density_estimator
      self.bgm_gmm_covariance = bgm_gmm_covariance
      self.bgm_gmm_n_components = bgm_gmm_n_components
      self.train_latent_dicts = train_latent_dicts
      self.test_latent_dicts = test_latent_dicts
      self.random_seed = random_seed
    
    def load_latent_codes(self):
      latent_codes_train_ID = self.train_latent_dicts[self.layer]['Train ID'].reshape(self.train_latent_dicts[self.layer]['Train ID'].shape[0],-1)
      latent_codes_train_OOD = self.train_latent_dicts[self.layer]['Train OOD'].reshape(self.train_latent_dicts[self.layer]['Train OOD'].shape[0],-1)
      latent_codes_test_ID = self.test_latent_dicts[self.layer]['Test ID'].reshape(self.test_latent_dicts[self.layer]['Test ID'].shape[0],-1)
      latent_codes_test_OOD = self.test_latent_dicts[self.layer]['Test OOD'].reshape(self.test_latent_dicts[self.layer]['Test OOD'].shape[0],-1)

      return latent_codes_train_ID, latent_codes_train_OOD, latent_codes_test_ID, latent_codes_test_OOD  

    def compute_MD(self,
      embedding: np.ndarray, # embedding of dimension (n_sample, n_dim)
      means: np.ndarray, # A matrix of size (num_classes, n_dim), where the ith row corresponds to the mean of the fitted Gaussian distribution for the i-th class.
      covariances: np.ndarray # The covariance of each mixture component. 
      ) -> np.ndarray: # A matrix of size (n_sample, n_class) where the (i, j) element corresponds to the Mahalanobis distance between i-th sample to the j-th class Gaussian.
      
      """Computes Mahalanobis distance between the input and the fitted Guassians. The Mahalanobis distance (Mahalanobis, 1936) is defined as
      $$distance(x, mu, sigma) = sqrt((x-\mu)^T \sigma^{-1} (x-\mu))$$ where `x` is a vector, `mu` is the mean vector for a Gaussian, and `sigma` 
      is the covariance matrix. We compute the distance for all examples in `embedding`, and across all classes in `means`.
      Note that this function technically computes the squared Mahalanobis distance."""

      if self.bgm_gmm_covariance.lower() == 'diag': 
        md = np.stack( [((embedding - means[j,:])**2/covariances[j,:]).sum(axis=1) for j in range(len(means))] ).T**(0.5)

      elif self.bgm_gmm_covariance.lower() == 'spherical': 
        md = np.stack([((embedding - means[j,:])**2).sum(axis=1)/covariances[j] for j in range(len(means))]).T**(0.5) 

      elif self.bgm_gmm_covariance.lower() == 'tied': 
        y_mu = np.stack([embedding - means[j,:] for j in range(len(means))])
        inv_covmat = np.linalg.inv(covariances)
        left = np.stack([np.dot(y_mu[j,:], inv_covmat) for j in range(len(means))])
        maha = np.stack([np.dot(left[j, :], y_mu[j,:].T) for j in range(len(means))])
        md = np.stack([maha[j,:].diagonal() for j in range(len(means))]).T**(0.5)
      return md

    def compute_RMD(self): 
      #generate latent 
      latent_codes_train_ID, latent_codes_train_OOD, latent_codes_test_ID, latent_codes_test_OOD  = self.load_latent_codes()

      #estimator 
      if self.density_estimator.lower() == 'bgm':  
        estimator = BGM(weight_concentration_prior_type="dirichlet_process", covariance_type = self.bgm_gmm_covariance, 
                n_components=self.bgm_gmm_n_components, reg_covar=1e-4, init_params='kmeans',
                max_iter=1000, mean_precision_prior=.5, random_state=2)
        estimator_supercluster = BGM(weight_concentration_prior_type="dirichlet_process", covariance_type = self.bgm_gmm_covariance, 
        n_components=1, reg_covar=1e-4, init_params='kmeans',
        max_iter=1000, mean_precision_prior=.5, random_state=2) #1 cluster 
      elif self.density_estimator.lower() == 'gmm': 
        estimator = GMM(n_components=self.bgm_gmm_n_components, init_params='kmeans', covariance_type=self.bgm_gmm_covariance, reg_covar=1e-4, random_state=2)  
        estimator_supercluster = GMM(n_components=1, init_params='kmeans', covariance_type=self.bgm_gmm_covariance, reg_covar=1e-4, random_state=2)  #1 cluster
      
      #fit estimator
      estimator = estimator.fit(latent_codes_train_ID)
      estimator_supercluster = estimator_supercluster.fit(latent_codes_train_ID)

      #calculate minimum MD to clusters  
      md_train_ID = self.compute_MD(latent_codes_train_ID, estimator.means_, estimator.covariances_).min(axis=1)
      md_train_OOD = self.compute_MD(latent_codes_train_OOD, estimator.means_, estimator.covariances_).min(axis=1)
      md_test_ID = self.compute_MD(latent_codes_test_ID, estimator.means_, estimator.covariances_).min(axis=1)
      md_test_OOD = self.compute_MD(latent_codes_test_OOD, estimator.means_, estimator.covariances_).min(axis=1)

      #calculate minimum MD to supercluster
      md_train_ID_1 = self.compute_MD(latent_codes_train_ID, estimator_supercluster.means_, estimator.covariances_).min(axis=1)
      md_train_OOD_1 = self.compute_MD(latent_codes_train_OOD, estimator_supercluster.means_, estimator.covariances_).min(axis=1)
      md_test_ID_1 = self.compute_MD(latent_codes_test_ID, estimator_supercluster.means_, estimator.covariances_).min(axis=1)
      md_test_OOD_1 = self.compute_MD(latent_codes_test_OOD, estimator_supercluster.means_, estimator.covariances_).min(axis=1)

      # #calculate RMD
      rmd_train_ID = md_train_ID - md_train_ID_1
      rmd_train_OOD = md_train_OOD - md_train_OOD_1
      rmd_test_ID = md_test_ID - md_test_ID_1
      rmd_test_OOD = md_test_OOD - md_test_OOD_1

      return rmd_train_ID, rmd_train_OOD, rmd_test_ID, rmd_test_OOD