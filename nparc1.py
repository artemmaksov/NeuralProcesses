import numpy as np
import tensorflow as tf

#encoder architecture 
def h_encoder(inputs, dim_h_hidden, dim_r):
    l1 = tf.layers.dense(inputs= inputs, units=dim_h_hidden, activation='sigmoid', name="encoder_layer1", reuse=tf.AUTO_REUSE)
    l2 = tf.layers.dense(inputs = l1, units=dim_r, name="encoder_layer2", reuse=tf.AUTO_REUSE)
    
    return l2
	
#aggregator for r ~ average 
def aggregate_r(inputs):
    l = tf.reduce_mean(inputs, axis=0)
    l2 = tf.reshape(l, shape=(1, -1))
    return l2
	
	
#decoder g 
def decoder_g(z_sample, x_star, dim_g_hidden, noise_sd = 0.05):
    
    n_draws = z_sample.get_shape()[0]
    N_star = tf.shape(x_star)[0]
    
    #z_sample_rep [n_draws, N_star, dim_z]
    z_sample_rep = tf.expand_dims(z_sample, axis=1)
    z_sample_rep = tf.tile(z_sample_rep, multiples=(1, N_star, 1))
    
    #x_star_rep [n_draws, N_star, dim_x]
    x_star_rep = tf.expand_dims(x_star, axis=0)
    x_star_rep = tf.tile(x_star_rep, multiples=(n_draws, 1, 1))
    
    #concatenate x_T and z
    inputs = tf.concat([x_star_rep, z_sample_rep], axis=2 )
    
    #hidden layer
    hidden = tf.layers.dense(inputs=inputs, units=dim_g_hidden, activation='sigmoid', name = "decoder_layer1", reuse=tf.AUTO_REUSE)
    
    #mu_star [N_star, n_draws]
    mu_star = tf.layers.dense(hidden, units=1, name = 'decoder_layer2_mu', reuse=tf.AUTO_REUSE)
    mu_star = tf.squeeze(mu_star, axis=2)
    mu_star = tf.transpose(mu_star)
    
    #assume fixed sigma
    sigma_star = tf.constant(noise_sd, dtype=tf.float32)

    return mu_star, sigma_star 