# Out-of-distribution detection for Pest Management 

* Affiliation: Harvard University, Institute for Applied Computational Science
* Authors: Austin Nguyen, Erin Tomlinson, Eric Hemold, Aloysius Lim, Molly Liu 

We introduce a repository for out-of-distribution (OOD) detection in partnership with Wadhwani AI.

# Overview  

Wadhwani AI is an independent nonprofit institute developing AI-based solutions for underserved communities in developing countries. They build and deploy AI solutions in partnership with local governments and civil society organizations to improve large-scale public programs. One such partnership is in the area of pest management for cotton farms. Cotton is the most important fiber and a cash crop for India, providing about 6 million farmers with a direct livelihood and 40-50 million people work in the cotton trade. Small-holder farmers, contributing 75% of the production, struggle with uncertainty in yield and income. Cotton is exceptionally vulnerable to pest attacks, with bollworms responsible for an estimated 70% of all pest damage. Bollworms destroy the seed coats of the plant which turns into harvestable cotton, despite heavy pesticide usage. While there have been some advances in GMO cotton called Bt cotton which naturally produces pyrethroids that repel insects, in many regions of the world, bollworms have developed resistance to these chemicals. Wadhwani AI has developed a mobile phone application called CottonAce that helps cotton farmers identify bollworms in their field. Bollworms are a pernicious pest for cotton farmers across the world, requiring consistent monitoring and expert decision making to properly address.  The app provides such support by recognizing bollworms in photos, then making recommendations based on what is found.

Our project goal is to identify and implement one or more effective solutions to the problem of out-of-distribution image detection, allowing the app to reject errant images with minimal processing overhead. Searching for a solution that is deployable in resource constrained environments i.e. without internet connectivity and with limited computing resources, is an important aspect of our task in addition to model accuracy since the end users are often in remote locations and using devices that would be inadequate to run computationally prohibitive models. Therefore, another key part of our modeling analysis will be to remain cognizant of the deployment constraints of our target audience and look for solutions that minimize the computational resources required to run them.

# Data Preparation 

* We sorted the data into in-distribution (ID), edge case (EC), and out-of-distribution (OOD). We define these categories as follows etc etc 

# Usage 

## Convolutional autoencoder (CAE)

Training

Evaluation 

## Bayesian Mixture Model & Relative Mahalanobis Distance


# License 

MIT Licnese

# Acknowledgements 

We thank Weiwei Pan for her mentorship and Wadhwani AI team for their partnership and support. 


