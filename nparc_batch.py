"""
This module defines Neural Processes architecture.
"""

import tensorflow as tf

#encoder architecture
def np_encoder(x_data, y_data, dim_h_hidden, dim_r, dim_z, act_f):
    """
    Encoder architecture.
    """
    inputs = tf.concat([x_data, y_data], axis=2)#, name = "NP_xy")

    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        el1 = tf.layers.dense(inputs=inputs, units=dim_h_hidden[0],
                              activation=act_f, name="encoder_layer_0", reuse=tf.AUTO_REUSE)

        if len(dim_h_hidden) > 1:
            for i, size in enumerate(dim_h_hidden[1:]):
                el1 = tf.layers.dense(inputs=el1, units=size,
                                      activation=act_f, name="encoder_layer_{}".format(i+1))

        el2 = tf.layers.dense(inputs=el1, units=dim_r, name="encoder_last", reuse=tf.AUTO_REUSE)

        r_agg = tf.reduce_mean(el2, axis=1)

        z_mu = tf.layers.dense(inputs=r_agg, units=dim_z, name="z_mu", reuse=tf.AUTO_REUSE)
        z_sigma = tf.layers.dense(inputs=r_agg, units=dim_z, name="z_sigma", reuse=tf.AUTO_REUSE)
        z_sigma = tf.nn.softplus(z_sigma)

    return z_mu, z_sigma

#decoder architecture
def decoder_g(z_sample, x_star, dim_g_hidden, act_f, noise_sd=0.05):
    """
    Decoder architecture.
    """
    n_draws = z_sample.get_shape()[0]
    n_star = tf.shape(x_star)[1]
    #batch_size = tf.shape(x_star)[0]

    #z_sample_rep [n_draws, n_star, dim_z]
    z_sample_rep = tf.expand_dims(z_sample, axis=2)
    z_sample_rep = tf.tile(z_sample_rep, multiples=(1, 1, n_star, 1))

    #x_star_rep [n_draws, N_star, dim_x]
    x_star_rep = tf.expand_dims(x_star, axis=0)
    x_star_rep = tf.tile(x_star_rep, multiples=(n_draws, 1, 1, 1))

    #concatenate x_T and z
    inputs = tf.concat([x_star_rep, z_sample_rep], axis=-1)

    #hidden layer
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(inputs=inputs, units=dim_g_hidden[0],
                                 activation=act_f, name="decoder_layer_0", reuse=tf.AUTO_REUSE)
        if len(dim_g_hidden) > 1:
            for i, size in enumerate(dim_g_hidden[1:]):
                hidden = tf.layers.dense(inputs=hidden, units=size,
                                         activation=act_f, name="decoder_layer_{}".format(i+1))

        mu_star = tf.layers.dense(hidden, units=1, name='decoder_last', reuse=tf.AUTO_REUSE)

    mu_star = tf.transpose(mu_star, perm=[3, 1, 2, 0])
    #mu_star, sigma_star = tf.split(mu_star, 2, axis=0)
    mu_star = tf.squeeze(mu_star, axis=0)
    #assume fixed sigma
    sigma_star = tf.constant(noise_sd, dtype=tf.float32)
    #sigma_star = tf.squeeze(sigma_star, axis=0)
    #sigma_star = noise_sd + (1 - noise_sd) * tf.nn.softplus(sigma_star)

    return mu_star, sigma_star
