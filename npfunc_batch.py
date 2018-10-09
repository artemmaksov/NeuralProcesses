import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from nparc_batch import decoder_g, np_encoder

#following the R code by Kaspar Martens https://github.com/kasparmartens/NeuralProcesses
	
#compute the probability of y_star data belonging to 
#a gaussian with y_pred_mu center and y_pred sigma scale 
def loglikelihood1(y_star, y_pred_mu, y_pred_sigma):
    
    reps = y_pred_mu.shape[1] 
    y_all = tf.tile(y_star, multiples=(1, reps))
    #loglik = tf.square(y_all - y_pred_mu)/tf.exp(y_pred_sigma)
    loglik = tf.exp(-tf.square(y_all - y_pred_mu)/(2*tf.square(y_pred_sigma))) #/ (y_pred_sigma * np.sqrt(2.0 * np.pi) )
    loglik = tf.reduce_sum(loglik, axis=0)
    loglik = tf.reduce_mean(loglik)
    return -0.5 * loglik
    #return loglik 
    
#same but the way it is done in Kaspar Martens code 
def loglikelihood2(y_star, y_pred_mu, y_pred_sigma):
    
    #p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=tf.square(y_pred_sigma))
    p_normal = tfp.distributions.Normal(loc = y_pred_mu, scale=y_pred_sigma)
    #p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=y_pred_sigma) 
    #p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=tf.exp(y_pred_sigma)) 	
    loglik = p_normal.log_prob(y_star)
    #loglik = tf.reduce_mean(loglik, axis=0) #changed to mean from sum to adjust for getting new data
    #loglik = tf.reduce_sum(loglik, axis=0)
    loglik = tf.reduce_mean(loglik)
    return -loglik
	
def loglikelihood3(y_star, y_pred_mu, y_pred_sigma):
    
    #p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=tf.square(y_pred_sigma))
    #p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=y_pred_sigma) 
    #p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=tf.exp(y_pred_sigma)) 	
    #loglik = p_normal.log_prob(y_star)
    #loglik = tf.reduce_mean(loglik, axis=0) #changed to mean from sum to adjust for getting new data
    #loglik = tf.reduce_sum(loglik, axis=0)
    reps = y_pred_mu.shape[1] 
    y_all = tf.tile(y_star, multiples=(1, reps))
    loglik = tf.keras.losses.binary_crossentropy(y_all, y_pred_mu)
    loglik = tf.reduce_mean(loglik)
    return loglik
	
#KL divergence for q and p 
def KLqp_gauss(mu_q, sigma_q, mu_p, sigma_p):
    
    #temp = (tf.exp(sigma_q) + tf.square(mu_q - mu_p)) / tf.exp(sigma_p) - 1 + sigma_p - sigma_q
    #temp = tf.square(sigma_q)/tf.square(sigma_p) + tf.square(mu_q - mu_p)/tf.square(sigma_p) - 1.0 + tf.log( tf.square(sigma_q)/tf.square(sigma_p) )  
    sq2 = tf.exp(sigma_q) + 1e-16
    sp2 = tf.exp(sigma_p) + 1e-16
    temp = sq2/sp2 + tf.square(mu_q - mu_p)/sp2 + tf.log(sp2/sq2 ) - 1.0  

    KL = 0.5 * tf.reduce_mean(temp) #changed to mean from sum to adjust for getting new data
    #KL = 0.5 * tf.reduce_sum(temp)    
    return KL
	
def KLuni(mu, sigma): 
    """Univariate KL divergence"""
    KL = 0.5 * tf.reduce_mean( (tf.exp(sigma) + tf.square(mu) - 1. - sigma))
    return KL
	
def distBD(mu_q, sigma_q, mu_p, sigma_p):
    """Bhattacharyya distance"""
    sq2 = tf.square(sigma_q) + 1e-16
    sp2 = tf.square(sigma_p) + 1e-16
    term1 = 0.25 * tf.log(0.25 * ( sp2/sq2 + sq2/sp2 + 2) ) 
    term2 = 0.25 * (tf.square(mu_p - mu_q)/ (sp2 + sq2))  
    bd = term1 + term2
    bd = tf.reduce_mean(bd)
	
    return bd
	

#prediction
def posterior_predict(x, y, x_star_value, dim_h_hidden, dim_g_hidden, dim_r, dim_z, epsilon = None , n_draws = 1, act_f = tf.nn.sigmoid):
    """
    Predict posterior. 
	
    Takes in: 
    Observed data: x, y
    Prediction coordinates: x_star_value 
    Decoder parameters: dim_h_hidden, dim_g_hidden, dim_r, dim_z
    Epsilon matrix for reparametrization: epsilon 
    Number of draws: n_draws
    Activation function: act_f 
    
    Returns:
    Predicted posterior over x_star: y_star 
    """
    
    x_obs = tf.constant(x, dtype = tf.float32 )#, name = "NP_x_obs")
    y_obs = tf.constant(y, dtype = tf.float32 )#, name = "NP_y_obs")
    x_star = tf.constant(x_star_value, dtype = tf.float32 )#, name = "NP_x_star")
    
    bs = tf.shape(x_obs)[0]
	
    #z_mu, z_sigma = map_xy_to_z_params(x_obs, y_obs, dim_h_hidden, dim_r, dim_z, act_f)
    z_mu, z_sigma = np_encoder(x_obs, y_obs, dim_h_hidden, dim_r, dim_z, act_f)
	
    
    if( epsilon == None):
        epsilon = tf.random_normal(shape=(n_draws, bs, dim_z) )#, name = "NP_epsilon")
        
    z_sample = tf.add(tf.multiply(epsilon, z_sigma), z_mu ) #, name = "NP_z_sample")
    #z_sample = tf.add(z_sample, z_mu)
    
    y_star = decoder_g(z_sample, x_star, dim_g_hidden, act_f)
    
    return y_star 

#prediction from eps values 
def prior_predict(x_star, dim_g_hidden, dim_z, epsilon = None, n_draws = 1):
    N_star = x_star.shape[0]
    x_star_tf = tf.constant(x_star, dtype=tf.float32 )
    
    if( epsilon == None):
        epsilon = tf.random_normal(shape=(n_draws, dim_z))
        
        
    z_sample = epsilon
    
    y_star = decoder_g(z_sample, x_star, dim_g_hidden, act_f)
    
    return y_star 
	
	
	

#initialize NP
def init_NP(x_context, y_context, x_target, y_target, dim_h_hidden, dim_g_hidden, dim_r, dim_z, elbo, noise_sd = 0.05, lr = 0.001, act_f = tf.nn.sigmoid, epsilon_std=None, n_draws=7):
	""" 
	Initializa neural process architecture. 
	Takes in: 
	Context points: x_context, y_context 
	All data: x_target, y_target 
	Dimensions of the encoder and decoder: dim_h_hidden, dim_g_hidden, dim_r, dim_z
	Loss funciton: elbo 
	Expectation of noise in the data: noise_sd 
	Learning rate: lr 
	Activation function: act_f 
	Random draw std for reparametrization: epsilon_std
	Number of random draws: n_draws
	
	Returns: 
	Trainining operation: train_op
	Loss: loss 
	"""
	
	#Use encoder to get Z(mu, sigma) for both context and all points 
	z_context_mu, z_context_sigma = np_encoder(x_context, y_context, dim_h_hidden, dim_r, dim_z, act_f)
	z_all_mu, z_all_sigma = np_encoder(x_target, y_target, dim_h_hidden, dim_r, dim_z, act_f)	
	
	#get the batch size 
	bs = tf.shape(x_context)[0]
	
    #sample z= mu + sigma*eps
	if epsilon_std == None :
		epsilon =  tf.random_normal(shape=(n_draws, bs, dim_z) , stddev = 1.0 )
	else:
		epsilon =  tf.random_normal(shape=(n_draws, bs, dim_z) , stddev = 1.0 + (epsilon_std - 1.0))

	z_sample = tf.add(  tf.multiply(epsilon, z_all_sigma), z_all_mu)
    
    #map (z, x_T) to y_T
	y_pred_mu, y_pred_sigma = decoder_g(z_sample, x_target, dim_g_hidden, act_f, noise_sd)
	
    #ELBO loss
	elboloss = elbo(y_target, y_pred_mu, y_pred_sigma)
	#KL = 1.0*KLqp_gauss(z_all_mu, z_all_sigma, z_context_mu, z_context_sigma) 
	KL = 1.0*distBD(z_all_mu, z_all_sigma, z_context_mu, z_context_sigma) 	
	loss = elboloss + KL 
    
    #optimization 
	optimizer = tf.train.AdamOptimizer(lr)
	train_op = optimizer.minimize(loss)
    
	return(train_op, loss)
	
	

#define gaussian activation function 
def g_act(x):
	""" Gaussian activation function """ 
	return tf.exp(tf.negative( tf.square(x) ) )
    




