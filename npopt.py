import numpy as np
import tensorflow as tf

from npfunc import posterior_predict, prior_predict



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
    N = np.random.randint(low=2, high=x_data.shape[0]+1 ) 
    #x_c_id = np.random.randint(low=0, high=x_data.shape[0], size=N)
    x_c_id = np.random.choice(x_data.shape[0], size =N)
    x_c = x_data[x_c_id, :]
    y_c = y_data[x_c_id, :]
    #select target points 
    x_targ = np.delete(x_data, x_c_id , 0)
    y_targ = np.delete(y_data, x_c_id , 0)
    
    a = sess.run(train_op_and_loss, feed_dict={x_context: x_c, y_context: y_c, x_target:x_targ, y_target:y_targ})
    
    return a 
	
def sample_curves(x_data, y_data, x_t, dim_h_hidden, dim_g_hidden, 
                  dim_r, dim_z,sess, epsilon_std=None, n_draws=1, act_f=tf.nn.sigmoid):
    
    #generate epsilon for sampling 
    if epsilon_std == None :
        eps =  tf.random_normal(shape=(n_draws, dim_z) , stddev = 1.0 )
    else:
        eps =  tf.random_normal(shape=(n_draws, dim_z) , stddev = 1.0 + (epsilon_std - 1) * np.random.rand())
    
    #predict curves 
    predict_op = posterior_predict(x_data, y_data, x_t, dim_h_hidden, dim_g_hidden, dim_r, dim_z, epsilon=eps, n_draws=n_draws, act_f=act_f)
    y_star_mat = sess.run(predict_op)
    
    #compute mean and std
    y_star_mean = np.mean(y_star_mat[0], axis=1)
    y_star_sigma = np.std(y_star_mat[0], axis=1)
    
    return y_star_mat, y_star_mean, y_star_sigma



	
	


    




