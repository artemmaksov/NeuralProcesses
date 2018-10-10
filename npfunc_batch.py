"""
This module contains:
function for initialization of neural processes,
functions for posterior and prior sampling,
losses.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from nparc_batch import decoder_g, np_encoder



#compute the probability of y_star data belonging to
#a gaussian with y_pred_mu center and y_pred sigma scale
def loglikelihood2(y_star, y_pred_mu, y_pred_sigma):
    """Compute loglikelihood given data y_pred and distribution N(y_pred_mu, y_pred_sigma)"""
    p_normal = tfp.distributions.Normal(loc=y_pred_mu, scale=y_pred_sigma)
    loglik = p_normal.log_prob(y_star)
    loglik = tf.reduce_mean(loglik)
    return -loglik

def loglikelihood3(y_star, y_pred_mu, *args):
    """Compute loss given data y_pred and predictions y_pred_mu"""
    reps = y_pred_mu.shape[1]
    y_all = tf.tile(y_star, multiples=(1, reps))
    loglik = tf.keras.losses.binary_crossentropy(y_all, y_pred_mu)
    loglik = tf.reduce_mean(loglik)
    return loglik

#KL divergence for q and p
def kl_qp_gauss(mu_q, sigma_q, mu_p, sigma_p):
    """Compute KL divergence between q and p"""
    #temp = (tf.exp(sigma_q) + tf.square(mu_q - mu_p)) / tf.exp(sigma_p) - 1 + sigma_p - sigma_q
    #temp = tf.square(sigma_q)/tf.square(sigma_p) + tf.square(mu_q - mu_p)/tf.square(sigma_p)
    #- 1.0 + tf.log( tf.square(sigma_q)/tf.square(sigma_p) )
    sq2 = tf.exp(sigma_q) + 1e-16
    sp2 = tf.exp(sigma_p) + 1e-16
    temp = sq2/sp2 + tf.square(mu_q - mu_p)/sp2 + tf.log(sp2/sq2) - 1.0

    kld = 0.5 * tf.reduce_mean(temp)#changed to mean from sum to adjust for getting new data
    return kld

def kl_uni(mu_q, sigma):
    """Univariate KL divergence"""
    kld = 0.5 * tf.reduce_mean((tf.exp(sigma) + tf.square(mu_q) - 1. - sigma))
    return kld

def dist_bd(mu_q, sigma_q, mu_p, sigma_p):
    """Bhattacharyya distance"""
    sq2 = tf.square(sigma_q) + 1e-16
    sp2 = tf.square(sigma_p) + 1e-16
    term1 = 0.25 * tf.log(0.25 * (sp2/sq2 + sq2/sp2 + 2))
    term2 = 0.25 * (tf.square(mu_p - mu_q)/ (sp2 + sq2))
    bdv = term1 + term2
    bdv = tf.reduce_mean(bdv)

    return bdv


#prediction
def posterior_predict(x_data, y_data, x_star_value, dim_h_hidden, dim_g_hidden, dim_r, dim_z,
                      epsilon=None, n_draws=1, act_f=tf.nn.sigmoid):
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

    x_obs = tf.constant(x_data, dtype=tf.float32)#, name = "NP_x_obs")
    y_obs = tf.constant(y_data, dtype=tf.float32)#, name = "NP_y_obs")
    x_star = tf.constant(x_star_value, dtype=tf.float32)#, name = "NP_x_star")

    batch_size = tf.shape(x_obs)[0]

    #z_mu, z_sigma = map_xy_to_z_params(x_obs, y_obs, dim_h_hidden, dim_r, dim_z, act_f)
    z_mu, z_sigma = np_encoder(x_obs, y_obs, dim_h_hidden, dim_r, dim_z, act_f)


    if epsilon is None:
        epsilon = tf.random_normal(shape=(n_draws, batch_size, dim_z))#, name = "NP_epsilon")

    z_sample = tf.add(tf.multiply(epsilon, z_sigma), z_mu) #, name = "NP_z_sample")

    y_star = decoder_g(z_sample, x_star, dim_g_hidden, act_f)

    return y_star

#prediction from eps values
def prior_predict(x_star, dim_g_hidden, dim_z, epsilon=None, n_draws=1, act_f=tf.nn.sigmoid):
    """
    Predict prior.
    """
    #N_star = x_star.shape[0]
    #x_star_tf = tf.constant(x_star, dtype=tf.float32)

    if epsilon is None:
        epsilon = tf.random_normal(shape=(n_draws, dim_z))

    z_sample = epsilon

    y_star = decoder_g(z_sample, x_star, dim_g_hidden, act_f)

    return y_star

#initialize NP
def init_neural_process(x_context, y_context, x_target, y_target, dim_h_hidden, dim_g_hidden,
                        dim_r, dim_z, elbo, noise_sd=0.05, lrv=0.001, act_f=tf.nn.sigmoid,
                        epsilon_std=None, n_draws=7):
    """
	Initialize neural process architecture.

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
    z_context_mu, z_context_sigma = np_encoder(x_context, y_context,
                                               dim_h_hidden, dim_r, dim_z, act_f)
    z_all_mu, z_all_sigma = np_encoder(x_target, y_target, dim_h_hidden, dim_r, dim_z, act_f)

    #get the batch size
    batch_size = tf.shape(x_context)[0]

    #sample z= mu + sigma*eps
    if epsilon_std is None:
        stddev = 1.0
    else:
        stddev = 1.0 + (epsilon_std - 1.0)

    epsilon = tf.random_normal(shape=(n_draws, batch_size, dim_z), stddev=stddev)
    z_sample = tf.add(tf.multiply(epsilon, z_all_sigma), z_all_mu)

    #map (z, x_T) to y_T
    y_pred_mu, y_pred_sigma = decoder_g(z_sample, x_target, dim_g_hidden, act_f, noise_sd)

    #ELBO loss
    elboloss = elbo(y_target, y_pred_mu, y_pred_sigma)
	#KL = 1.0*KLqp_gauss(z_all_mu, z_all_sigma, z_context_mu, z_context_sigma)
    kld = 1.0*dist_bd(z_all_mu, z_all_sigma, z_context_mu, z_context_sigma)
    loss = elboloss + kld

    #optimization
    optimizer = tf.train.AdamOptimizer(lrv)
    train_op = optimizer.minimize(loss)

    return(train_op, loss)

#define gaussian activation function
def g_act(xnw):
    """ Gaussian activation function """
    return tf.exp(tf.negative(tf.square(xnw)))
