import numpy as np
import tensorflow as tf
from nparc1 import h_encoder, aggregate_r, decoder_g, np_encoder

#following the R code by Kaspar Martens https://github.com/kasparmartens/NeuralProcesses


#return Z sampling from input r 
# def get_z_params(inputs_r, dim_z):
    # mu = tf.layers.dense(inputs=inputs_r, units=dim_z, name="z_params_mu", reuse=tf.AUTO_REUSE)
    # sigma = tf.nn.softplus(tf.layers.dense(inputs=inputs_r, units=dim_z, name="z_params_sigma", reuse=tf.AUTO_REUSE))
    
    # return mu, sigma
	
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
    p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=y_pred_sigma) 
    #p_normal = tf.distributions.Normal(loc= y_pred_mu, scale=tf.exp(y_pred_sigma)) 	
    loglik = p_normal.log_prob(y_star)
    loglik = tf.reduce_mean(loglik, axis=0) #changed to mean from sum to adjust for getting new data
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
    KL = 0.5 * tf.reduce_mean( (tf.exp(sigma) + tf.square(mu) - 1. - sigma))
    return KL
	
def distBD(mu_q, sigma_q, mu_p, sigma_p):

    sq2 = tf.square(sigma_q) + 1e-16
    sp2 = tf.square(sigma_p) + 1e-16
    term1 = 0.25 * tf.log(0.25 * ( sp2/sq2 + sq2/sp2 + 2) ) 
    term2 = 0.25 * (tf.square(mu_p - mu_q)/ (sp2 + sq2))  
    bd = term1 + term2
    bd = tf.reduce_mean(bd)
	
    return bd
	
#mapping xy to z
# def map_xy_to_z_params(x, y, dim_h_hidden, dim_r, dim_z, act_f):
    
    # xy = tf.concat([x, y], axis= 1 )#, name = "NP_xy")
    # h = h_encoder(xy, dim_h_hidden, dim_r, act_f)
    # r = aggregate_r(h)
    # z_mu, z_sigma = get_z_params(r, dim_z)
    
    # return z_mu, z_sigma	
	

#prediction
def posterior_predict(x, y, x_star_value, dim_h_hidden, dim_g_hidden, dim_r, dim_z, epsilon = None , n_draws = 1, act_f = tf.nn.sigmoid):
    x_obs = tf.constant(x, dtype = tf.float32 )#, name = "NP_x_obs")
    y_obs = tf.constant(y, dtype = tf.float32 )#, name = "NP_y_obs")
    x_star = tf.constant(x_star_value, dtype = tf.float32 )#, name = "NP_x_star")
    
    #z_mu, z_sigma = map_xy_to_z_params(x_obs, y_obs, dim_h_hidden, dim_r, dim_z, act_f)
    z_mu, z_sigma = np_encoder(x_obs, y_obs, dim_h_hidden, dim_r, dim_z, act_f)
	
    
    if( epsilon == None):
        epsilon = tf.random_normal(shape=(n_draws, dim_z) )#, name = "NP_epsilon")
        
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
def init_NP(x_context, y_context, x_target, y_target, dim_h_hidden, dim_g_hidden, dim_r, dim_z, elbo, noise_sd = 0.05, lr = 0.001, act_f = tf.nn.sigmoid):
    
    #concatenate context and target
	x_all = tf.concat( [x_context, x_target], axis=0) #, name = "NP_x_all" )
	y_all = tf.concat( [y_context, y_target], axis=0) #, name = "NP_y_all")
    
    #map input to z
	#z_context_mu, z_context_sigma = map_xy_to_z_params(x_context, y_context, dim_h_hidden, dim_r, dim_z, act_f)
	#z_all_mu, z_all_sigma = map_xy_to_z_params(x_all, y_all, dim_h_hidden, dim_r, dim_z, act_f)
	
	z_context_mu, z_context_sigma = np_encoder(x_context, y_context, dim_h_hidden, dim_r, dim_z, act_f)
	z_all_mu, z_all_sigma = np_encoder(x_all, y_all, dim_h_hidden, dim_r, dim_z, act_f)	
    
    #sample z= mu + sigma*eps
	epsilon = tf.random_normal(shape=(7, dim_z))#, mean=0.0, stddev=1.0)    
	z_sample = tf.add(  tf.multiply(epsilon, z_all_sigma), z_all_mu)
    
    #map (z, x_T) to y_T
	y_pred_mu, y_pred_sigma = decoder_g(z_sample, x_target, dim_g_hidden, act_f, noise_sd)
	
    #ELBO
	#loglik = loglikelihood1(y_target, y_pred_mu, y_pred_sigma)
	elboloss = elbo(y_target, y_pred_mu, y_pred_sigma)
	#KL = 1.0*KLqp_gauss(z_all_mu, z_all_sigma, z_context_mu, z_context_sigma) 
	#KL2 = 0.1*KLuni(z_all_mu, z_all_sigma)
	KL = 1.0*distBD(z_all_mu, z_all_sigma, z_context_mu, z_context_sigma) 	
	#KL2 = 1.0*KLqp_gauss(y_pred_mu, noise_sd, y_pred_mu, y_pred_sigma) 
	loss = elboloss + KL #+ KL2
    
    
    #optimization 
	optimizer = tf.train.AdamOptimizer(lr)
	train_op = optimizer.minimize(loss)
    
	return(train_op, loss)
	
	


    




