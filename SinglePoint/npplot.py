import numpy as np
import matplotlib 
import matplotlib.pylab as plt
import seaborn 

def plot_np_fit_1D(x_t, y_t, x_data, y_data, y_star_mean, y_star_sigma, u_scale=1.0, ylim = None, lloc = "lower right", fignum=101):

	fig = plt.figure(fignum)
	plt.plot(x_t, y_t, 'r:', label=u'$f(x) = \sin(x)$')
	plt.plot(x_data[0:-2], y_data[0:-2], 'r.', markersize=10, label=u'Observations')
	plt.plot(x_t, y_star_mean, 'b-')
	
	plt.fill(np.concatenate([x_t, x_t[::-1]]),
		np.concatenate([y_star_mean - u_scale*1.9600 * y_star_sigma,(y_star_mean + u_scale*1.9600 * y_star_sigma  )[::-1]]),
			alpha=.5, fc='b', ec='None', label='Uncertainty')
			
	plt.plot(x_t[np.argmax(y_star_mean)], np.amax(y_star_mean), 'gx', markersize=20, label=u'Maximum')
	plt.plot(x_data[-2], y_data[-2], 'g.', markersize=20, label=u'Latest evaluation')
	plt.plot(x_data[-1], y_data[-1], 'k.', markersize=20, label=u'Next sample')
	plt.xlabel('$x$')
	plt.ylabel('$f(x)$')
	plt.ylim( ylim )
	plt.legend(loc= lloc)
	return fig 
	






	
	


    




