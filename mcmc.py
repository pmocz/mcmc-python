import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Metropolis-Hastings Markov Chain Monte Carlo Algorithm for Bayesian Inference (With Python)
Philip Mocz (2023), @PMocz

Apply Markov Chain Monte Carlo to fit exoplanet radial velocity data and
estimate the posterior distribution of the model parameters

"""


def rv_model( V, K, w, e, P, chi, t ):
	"""
    Calculate the radial velocity curve of exoplanet system
    V      systemic velocity
    K      velocity semiamplitude
    w      longitude of periastron
    e      eccentricity
    P      orbital period
    chi    fraction of an orbit, prior to t=0, at which periastron occured
    t      array of times to output radial velocities
	"""
	# initialize RV curve
	rv = 0*t + V

		
	# 1. calculate mean anomaly M (in [0,2pi])
	M = ( 2.0*np.pi / P ) * ( t + chi*P )
	M = M % (2.0*np.pi)
	
	# 2. calculate eccentric anomaly E
	#    by solving Kepler's equation with Newton-Raphson iterator
	E = 0*t
	tolerance = 1.0e-8
	max_iter = 100
	for i in range(len(t)):
		Ei = np.pi
		# f = @(E) E - e*sin(E) - M[i]
		# f_prime = @(E) = 1 - e*np.cos(E)
		it = 0
		dE = 1
		while (it <= max_iter) and (np.abs(dE) > tolerance):
			dE = -(Ei - e*np.sin(Ei) - M[i])/(1.0 - e*np.cos(Ei))   # dE = -f(Ei)/f_prime(Ei)
			Ei += dE
			it += 1

		if np.abs(dE) > tolerance:
			print('Error: Newton-Raphson Iterative failed!')

		E[i] = Ei

	# 3. calculate true anomaly f 
	#    (http://en.wikipedia.org/wiki/True_anomaly)
	f = 2.0*np.arctan2( np.sqrt(1.0+e)*np.sin(E/2.0), np.sqrt(1.0-e)*np.cos(E/2.0) )

	# 4. add effect of the planet to the RV curve
	rv -= K * ( np.cos(f+w) + e*np.cos(w) )

	return rv

def log_prior(theta, theta_lo, theta_hi):
	"""
    Calculate the log of the priors for a set of parameters 'theta'
    We assume uniform priors bounded by 'theta_lo' and 'theta_hi'
	"""
	return -np.sum( np.log(theta_hi - theta_lo))
	
def propose(theta_prev, sigma_theta, theta_lo, theta_hi):
	"""
    propose a new set of parameters 'theta' given the previous value
    'theta_prev' in the Markov chain. Choose new values by adding a 
    random Gaussian peturbation with standard deviation 'sigma_theta'.
    Make sure the new proposed value is bounded between 'theta_lo'
    and 'theta_hi'
	"""
	# propose a set of parameters
	theta_prop = np.random.normal(theta_prev, sigma_theta)
	
	# reflect proposals outside of bounds
	too_hi = theta_prop > theta_hi
	too_lo = theta_prop < theta_lo

	theta_prop[too_hi] = 2*theta_hi[too_hi] - theta_prop[too_hi]
	theta_prop[too_lo] = 2*theta_lo[too_lo] - theta_prop[too_lo]
	
	return theta_prop


def eval_model(theta, t):
	"""
    Evaluate the RV model given parameters 'theta' at times 't'
	"""
	V = theta[0]
	K = theta[2]
	w = theta[3]
	e = theta[4]
	P = theta[5]
	chi = theta[6]
	return rv_model( V, K, w, e, P, chi, t )


def log_likelihood(rv_pred, s, rv_data, rv_errors):
	"""
    Evaluate the log likelihood of a model 'rv_pred' given the data
	"""
	return np.sum( np.log(1.0/np.sqrt(2.0*np.pi*(rv_errors**2+s**2))) + (-(rv_pred-rv_data)**2 / (2*(rv_errors**2+s**2))) )


def log_posterior(theta, t, rv_data, rv_errors):
	"""
    Evaluate the log posterior of a model parameters 'theta' given the data
    Note: since our priors are constant, we ignore adding it
	"""
	s = theta[1]
	rv_pred = eval_model(theta, t)
	return log_likelihood(rv_pred, s, rv_data, rv_errors)


def main():
	""" Fit Radial Velocity curve parameters with MCMC """
	
	# set the random number generator seed
	np.random.seed(917)
	
	# Generate Mock Data
	N_params = 7
	V   = 0.0     # systemic velocity
	s   = 1.0     # stellar jitter (Gaussian error)
	K   = 40.0    # velocity semiamplitude
	w   = 4.2     # longitude of periastron
	e   = 0.3     # eccentricity
	P   = 45.0    # orbital period
	chi = 0.4     # fraction of an orbit, prior to t=0, at which periastron occured
	
	t_max = 100.0
	t  = np.linspace(0, t_max,  30)  # array of times
	tt = np.linspace(0, t_max, 200)  # dense array of times
	
	# array of Gaussian measurement errors at each time
	rv_errors = 2.0 + np.abs(np.random.normal(0.0, 1.0, size=t.size))	
	
	# exact radial velocity curve, for plotting
	rv_exact = rv_model( V, K, w, e, P, chi, tt)
	
	# mock data set (30 points)
	rv_data  = rv_model( V, K, w, e, P, chi, t) 
	for i in range(len(t)):
		rv_data[i] += np.random.normal(0.0,np.sqrt(s**2 + rv_errors[i]**2))
	
	# Bounds on Priors
	V_bounds   = np.array([-4.0,      4.0])
	s_bounds   = np.array([0.5,       1.2])
	K_bounds   = np.array([20.0,     60.0])
	w_bounds   = np.array([0.0, 2.0*np.pi])
	e_bounds   = np.array([0.0,       0.5])
	P_bounds   = np.array([30.0,     50.0])
	chi_bounds = np.array([0.0,       1.0])
	
	theta_bounds = np.array([V_bounds, s_bounds, K_bounds, w_bounds, e_bounds, P_bounds, chi_bounds])
	theta_lo = theta_bounds[:,0]
	theta_hi = theta_bounds[:,1]
	
	sigma_theta = 0.02 * (theta_hi - theta_lo)

	# prep figure
	fig = plt.figure(figsize=(6,4), dpi=80)
	
	# plot exact rv curve and mock data
	plt.plot(tt, rv_exact)
	plt.errorbar(t, rv_data, rv_errors, fmt='o')
	plt.xlabel("time [day]")
	plt.ylabel("radial velocity [m/s]")
	plt.xlim([-10,t_max+10])
	plt.ylim([-60, 60])
		
	# Carry out MCMC fitting to get best-fit parameters	
	Nburnin = 1000
	N = 8000 + Nburnin
	theta = np.zeros((N,N_params))
	
	theta_prev = np.random.uniform(theta_lo, theta_hi)
	
	for i in range(N):
		# take random step using the proposal distribution
		theta_prop = propose(theta_prev, sigma_theta, theta_lo, theta_hi)
		
		P_prop = log_posterior(theta_prop, t, rv_data, rv_errors)
		P_prev = log_posterior(theta_prev, t, rv_data, rv_errors)
		
		U = np.random.uniform(0.0, 1.0)
		r = np.min([1.0, np.exp(P_prop-P_prev)])
		
		if (U <= r):
			theta[i,:] = theta_prop
			theta_prev = theta_prop
		else:
			theta[i,:] = theta_prev

		# plot proposed function
		if (i % 100) == 0:
			print(theta[i,:])
			rv = eval_model(theta[i,:], tt)
			c = (1.0 - i/N)*0.5
			plt.plot(tt, rv, linewidth=0.5, color=(c,c,c))
			plt.pause(0.0001)
			
	# Save figure
	plt.savefig('mcmc.png',dpi=240)
	plt.show()
	
	# cut off burnin
	theta = theta[Nburnin:,:]
	
	# Plot Posteriors
	
	fig, ((ax0, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(6,4), dpi=80)
	
	n_bins = 20
	ax0.hist(theta[:,0], n_bins, histtype='step', fill=True)
	ax0.axvline(V, color='r', linewidth=1)
	ax0.set_title('V posterior')
	ax2.hist(theta[:,2], n_bins, histtype='step', fill=True)
	ax2.axvline(K, color='r', linewidth=1)
	ax2.set_title('K posterior')
	ax3.hist(theta[:,3], n_bins, histtype='step', fill=True)
	ax3.axvline(w, color='r', linewidth=1)
	ax3.set_title('w posterior')
	ax4.hist(theta[:,4], n_bins, histtype='step', fill=True)
	ax4.axvline(e, color='r', linewidth=1)
	ax4.set_title('e posterior')
	ax5.hist(theta[:,5], n_bins, histtype='step', fill=True)
	ax5.axvline(P, color='r', linewidth=1)
	ax5.set_title('P posterior')
	ax6.hist(theta[:,6], n_bins, histtype='step', fill=True)
	ax6.axvline(chi, color='r', linewidth=1)
	ax6.set_title('chi posterior')
	
	fig.tight_layout()
	
	
	# Save figure
	plt.savefig('mcmc2.png',dpi=240)
	plt.show()
	    
	return 0



if __name__== "__main__":
  main()

