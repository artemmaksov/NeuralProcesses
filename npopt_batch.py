"""
This module contains functions for optimization with Neural Processes.
"""
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from npfunc_batch import posterior_predict

def np_iteration(x_data, y_data, x_context, y_context, x_target, y_target, sess, train_op_and_loss):
    """Performs one iteration of Neural Process

    Takes in:
    Observed coordiantes and values: x_data, y_data
    Tensorflow placeholders: x_context, y_context, x_target, y_target
    Tensorflow session: sess
    Training operation and loss: train_op_and_loss

    Outputs: Training operation and loss a
    """

    #select N context points
    n_c = np.random.randint(low=2, high=x_data.shape[1]+1)
    x_c_id = np.random.choice(x_data.shape[1], size=n_c, replace=False)
    x_c = x_data[:, x_c_id, :]
    y_c = y_data[:, x_c_id, :]

	#run 1 training operation
    train = sess.run(train_op_and_loss, feed_dict={x_context: x_c, y_context: y_c, x_target:x_data, y_target:y_data})

    return train

def sample_curves(x_data, y_data, x_t, dim_h_hidden, dim_g_hidden,
                  dim_r, dim_z, sess, epsilon_std=None, n_draws=1, act_f=tf.nn.sigmoid):

    """ Sample the parameter space x_t using known coordinates x_data and values y_data.

    Takes in:
    Parameters of the NP model: dim_h_hidden, dim_g_hidden, dim_r, dim_z, act_f
    Session: sess
    Std of epsilon: epsilon_std
    Number of function samples to produce: n_draws

    Outputs:
    Sampled functions: y_star_mat
    Mean: y_star_mean
    STD: y_star_sigma
    """

    batch_size = tf.shape(x_data)[0]

    #generate epsilon for sampling
    if epsilon_std is None:
        stddev = 1.0
    else:
        stddev = 1.0 + (epsilon_std - 1)

    eps = tf.random_normal(shape=(n_draws, batch_size, dim_z), stddev=stddev)
	#predict curves
    predict_op = posterior_predict(x_data, y_data, x_t, dim_h_hidden, dim_g_hidden,
                                   dim_r, dim_z, epsilon=eps, n_draws=n_draws, act_f=act_f)
    y_star_mat = sess.run(predict_op)

    #compute mean and std
    y_star_mean = np.mean(y_star_mat[0], axis=-1)
    y_star_sigma = np.std(y_star_mat[0], axis=-1)

    return y_star_mat, y_star_mean, y_star_sigma

def expected_improvement(y_star_mean, y_star_sigma, y_data, y_predicted, ksi=0.01):
    """Compute the expected improvement over the sampling space.

    Takes in:
    Mean predicted curve and deviation: y_star_mean, y_star_sigma
    Observed points: y_data
    Predictions at observed points: y_predicted
    Ksi: exploration parameter

    Outputs:
    Expected improvement
    """

    fmax = np.amax(y_data - y_predicted)

    imp = y_star_mean - fmax - ksi
    imp_z = imp / (y_star_sigma + 1e-16)
    eiv = imp * norm.cdf(imp_z)  + y_star_sigma * norm.pdf(imp_z)

    return eiv

def get_next_sample(x_t, idx, fnc=None, y_t=None):
    """Gets the next data point.

    Takes in:
    Sample space: x_t
    Index of observation: idx
    Evaluation function: fn
    True values: y_t

    Outputs:
    New (x, y)
    """

    x_new = x_t[:, idx]
    if np.any(y_t):
        y_new = y_t[:, idx]

    else:
        y_new = fnc(x_new)

    return np.atleast_3d(x_new), np.atleast_3d(y_new)
