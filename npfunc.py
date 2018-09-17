import numpy as np
import tensorflow as tf
from nparc1 import h_encoder, aggregate_r, decoder_g

#following the R code by Kaspar Martens https://github.com/kasparmartens/NeuralProcesses


#return Z sampling from input r 
def get_z_params(inputs_r, dim_z):
    mu = tf.layers.dense(inputs=inputs_r, units=dim_z, name="z_params_mu", reuse=tf.AUTO_REUSE)
    sigma = tf.nn.softplus(tf.layers.dense(inputs=inputs_r, units=dim_z, name="z_params_sigma", reuse=tf.AUTO_REUSE))
    
    return mu, sigma
	
#compute the probability of y_star data belonging to 
#a gaussian with y_pred_mu center and y_pred sigma scale 
def loglikelihood1(y_star, y_pred_mu, y_pred_sigma):
    
    reps = y_pred_mu.shape[1] 
    y_all = tf.tile(y_star, multiples=(1, reps))
    
    loglik = tf.square(y_all - y_pred_mu)/tf.exp(y_pred_sigma)
    loglik = tf.reduce_sum(loglik, axis=0)
    loglik = tf.reduce_mean(loglik)
    return -0.5 * loglik
    
#same but the way it is done in Kaspar Martens code 
def loglikelihood2(y_star, y_pred_mu, y_pred_sigma):
    
    p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=y_pred_sigma)
    
    loglik = p_normal.log_prob(y_star)
    loglik = tf.reduce_sum(loglik, axis=0)
    loglik = tf.reduce_mean(loglik)
    return loglik
	
#KL divergence for q and p 
def KLqp_gauss(mu_q, sigma_q, mu_p, sigma_p):
    
    temp = (tf.exp(sigma_q) + tf.square(mu_q - mu_p)) / tf.exp(sigma_p) - 1 + sigma_p - sigma_q
    KL = 0.5 * tf.reduce_sum(temp)
    
    return KL
	
#mapping xy to z
def map_xy_to_z_params(x, y, dim_h_hidden, dim_r, dim_z):
    
    xy = tf.concat([x, y], axis= 1)
    h = h_encoder(xy, dim_h_hidden, dim_r)
    r = aggregate_r(h)
    z_mu, z_sigma = get_z_params(r, dim_z)
    
    return z_mu, z_sigma	
	

#prediction
def posterior_predict(x, y, x_star_value, dim_h_hidden, dim_g_hidden, dim_r, dim_z, epsilon = None , n_draws = 1):
    x_obs = tf.constant(x, dtype = tf.float32)
    y_obs = tf.constant(y, dtype = tf.float32)
    x_star = tf.constant(x_star_value, dtype = tf.float32)
    
    z_mu, z_sigma = map_xy_to_z_params(x_obs, y_obs, dim_h_hidden, dim_r, dim_z)
    
    if( epsilon == None):
        epsilon = tf.random_normal(shape=(n_draws, dim_z))
        
    z_sample = tf.multiply(epsilon, z_sigma)
    z_sample = tf.add(z_sample, z_mu)
    
    y_star = decoder_g(z_sample, x_star, dim_g_hidden)
    
    return y_star 

#prediction from eps values 
def prior_predict(x_star, dim_g_hidden, dim_z, epsilon = None, n_draws = 1):
    N_star = x_star.shape[0]
    x_star_tf = tf.constant(x_star, dtype=tf.float32 )
    
    if( epsilon == None):
        epsilon = tf.random_normal(shape=(n_draws, dim_z))
        
        
    z_sample = epsilon
    
    y_star = decoder_g(z_sample, x_star, dim_g_hidden)
    
    return y_star 
	
	
	

#initialize NP
def init_NP(x_context, y_context, x_target, y_target, dim_h_hidden, dim_g_hidden, dim_r, dim_z, lr = 0.001):
    
    #concatenate context and target
    x_all = tf.concat( [x_context, x_target], axis=0)
    y_all = tf.concat( [y_context, y_target], axis=0)
    
    #map input to z
    z_context_mu, z_context_sigma = map_xy_to_z_params(x_context, y_context, dim_h_hidden, dim_r, dim_z)
    z_all_mu, z_all_sigma = map_xy_to_z_params(x_all, y_all, dim_h_hidden, dim_r, dim_z)
    
    #sample z= mu + sigma*eps
    epsilon = tf.random_normal(shape=(7, dim_z))
        
    z_sample = tf.add(  tf.multiply(epsilon, z_all_sigma), z_all_mu)
    
    #map (z, x_T) to y_T
    y_pred_mu, y_pred_sigma = decoder_g(z_sample, x_target, dim_g_hidden)
    
    #ELBO
    loglik = loglikelihood1(y_target, y_pred_mu, y_pred_sigma)
    KL = KLqp_gauss(z_all_mu, z_all_sigma, z_context_mu, z_context_sigma) 
    loss = tf.negative(loglik) + KL
    
    
    #optimization 
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)
    
    return(train_op, loss)
	
	
	
#aggregator for r ~ average 
def aggregate_r(inputs):
    l = tf.reduce_mean(inputs, axis=0)
    l2 = tf.reshape(l, shape=(1, -1))
    return l2
	

	


    




