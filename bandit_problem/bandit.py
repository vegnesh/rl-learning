import numpy as np
import matplotlib.pyplot as plt

def bandit(k):
	return np.random.randn(k)

def eps_greedy(eps,q_a):
	my_eps = np.random.rand(1)
	if my_eps>eps:
		return np.argmax(q_a)
	else:
		return np.random.randint(0,len(q_a))


def bandit_problem(k=10,NT=1000,eps=0.1,initial_value=0):
	q_t = bandit(k)
	r_t = np.zeros(NT)
	q_a = np.zeros(k) + initial_value
	n_a = np.zeros(k)
	

	for ival in range(NT):
		index = eps_greedy(eps,q_a)
		n_a[index] = n_a[index] + 1
		rval       = np.random.randn(1) + q_t[index]
		q_a[index] = q_a[index] + (1.0/n_a[index])*(rval - q_a[index])
		r_t[ival]  = rval
	return r_t  
		


if __name__ == '__main__':
	print("Hello World")
	print('Running Bandit Problem')
	nruns   = 2000
	nsteps  = 1000
	nbandits= 10
	initial_bias = 0
	r_greedy     = np.zeros(nsteps)
	r_egreedy_01 = np.zeros(nsteps)
	r_egreedy_001= np.zeros(nsteps)
	r_optimistic = np.zeros(nsteps)
	for iters in range(nruns):
		r_greedy      += bandit_problem(nbandits,nsteps,0,initial_bias)
		r_egreedy_01  += bandit_problem(nbandits,nsteps,0.1,initial_bias)		
		r_egreedy_001 += bandit_problem(nbandits,nsteps,0.01,initial_bias)
		r_optimistic  += bandit_problem(nbandits,nsteps,0,10)

	r_greedy /= nruns
	r_egreedy_01 /= nruns
	r_egreedy_001 /= nruns
	r_optimistic /= nruns
	
	plt.plot(r_greedy,label = 'Greedy - 0 bias')
	plt.plot(r_egreedy_01,label ='Eps-Greedy - 0.1' )
	plt.plot(r_egreedy_001, label ='Eps-Greedy - 0.01' )
	plt.plot(r_optimistic, label = 'Greedy - 10 bias')
	plt.legend()
	plt.ylabel('Average Reward')
	plt.xlabel('Steps')
	
 	plt.show()
		
