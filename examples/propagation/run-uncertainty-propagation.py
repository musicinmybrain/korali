#!/usr/bin/env python3

# In this example, we sample the posterior of the parameters in a linear model.
# Next we evaluate the model for all the parameters in the sample database and
# different input data. Finally, the credible intervals can be plotted.

# Importing the computational model
import json
import sys
sys.path.append('./_model')
from work import *

data = {}
data['X'] = getReferencePoints()
data['Y'] = getReferenceData()

# Sample the posterior of the parameters
import korali
e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e['Problem']['Type'] = 'Bayesian/Reference'
e['Problem']['Likelihood Model'] = 'Normal'
e['Problem']['Reference Data'] = data['Y']

computational_model = lambda sampleData: model(sampleData, data['X'])
e['Problem']['Computational Model'] = computational_model

# Configuring TMCMC parameters
e['Solver']['Type'] = 'Sampler/TMCMC'
e['Solver']['Population Size'] = 1000
e['Solver']['Target Coefficient Of Variation'] = 0.8
e['Solver']['Covariance Scaling'] = 0.04

# Configuring the problem's random distributions
e['Distributions'][0]['Name'] = 'Uniform 0'
e['Distributions'][0]['Type'] = 'Univariate/Uniform'
e['Distributions'][0]['Minimum'] = -5.0
e['Distributions'][0]['Maximum'] = +5.0

e['Distributions'][1]['Name'] = 'Uniform 1'
e['Distributions'][1]['Type'] = 'Univariate/Uniform'
e['Distributions'][1]['Minimum'] = -5.0
e['Distributions'][1]['Maximum'] = +5.0

e['Distributions'][2]['Name'] = 'Uniform 2'
e['Distributions'][2]['Type'] = 'Univariate/Uniform'
e['Distributions'][2]['Minimum'] = 0.0
e['Distributions'][2]['Maximum'] = +5.0

# Configuring the problem's variables and their prior distributions
e['Variables'][0]['Name'] = 'a'
e['Variables'][0]['Prior Distribution'] = 'Uniform 0'

e['Variables'][1]['Name'] = 'b'
e['Variables'][1]['Prior Distribution'] = 'Uniform 1'

e['Variables'][2]['Name'] = '[Sigma]'
e['Variables'][2]['Prior Distribution'] = 'Uniform 2'

e['Store Sample Information'] = True

# Configuring output settings
e['File Output']['Path'] = '_korali_result_samples'

# Starting Korali's Engine and running experiment
e['Console Output']['Verbosity'] = 'Detailed'
k = korali.Engine()
k.run(e)

# Evaluate the model for all the parameters from the previous step
e = korali.Experiment()

x = np.linspace(0, 7, 100)

e['Problem']['Type'] = 'Propagation'
computational_model = lambda modelData: model_propagation(modelData, x)
e['Problem']['Execution Model'] = computational_model

# load the data from the sampling
with open('_korali_result_samples/latest') as f:
    d = json.load(f)

e['Variables'][0]['Name'] = 'a'
v = [p[0] for p in d['Results']['Posterior Sample Database']]
e['Variables'][0]['Precomputed Values'] = v

e['Variables'][1]['Name'] = 'b'
v = [p[1] for p in d['Results']['Posterior Sample Database']]
e['Variables'][1]['Precomputed Values'] = v

e['Variables'][2]['Name'] = 'sigma'
v = [p[2] for p in d['Results']['Posterior Sample Database']]
e['Variables'][2]['Precomputed Values'] = v

e['Solver']['Type'] = 'Executor'
e['Solver']['Executions Per Generation'] = 100

e['Console Output']['Verbosity'] = 'Minimal'
e['File Output']['Path'] = '_korali_result_propagation'
e['Store Sample Information'] = True

k = korali.Engine()
k.run(e)

# Uncomment the next two lines to plot the credible intervals
# from plots import *
# plot_credible_intervals('./_korali_result_propagation/latest', data)
