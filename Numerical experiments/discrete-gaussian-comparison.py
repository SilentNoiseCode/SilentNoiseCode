import math
import numpy as np
from fractions import Fraction

# We import the implementation of discrete gaussian distribution sampler proposed by C. Canonne, G. Kamath, and T. Steinke.
# See https://arxiv.org/abs/2004.00010 for details.
import discretegauss
import cdp2adp

eps = 1
delta = 1e-8
# convert to concentrated DP
rho=cdp2adp.cdp_rho(eps,delta)
print(str(rho)+"-CDP implies ("+str(eps)+","+str(delta)+")-DP")
sigma2=1/(2*rho)

N = 2**10 # repetition
n = 13
t = 3

print("semihonest:")
noise_semihonest = np.zeros(N)
m = n - t
for h in range(N):
    noise = 0
    for i in range(n):
        noise += discretegauss.sample_dgauss(sigma2/m)
    noise_semihonest[h] = noise
var_semihonest = np.var(noise_semihonest)
print("MSE is "+str(var_semihonest))

print("baseline:")
noise_baseline = np.zeros(N)
m = 1
for h in range(N):
    noise = 0
    for i in range(math.comb(n, t)):
        noise += discretegauss.sample_dgauss(sigma2/m)
    noise_baseline[h] = noise
var_baseline = np.var(noise_baseline)
print("MSE is "+str(var_baseline))

print("allsets:")
noise_allsets = np.zeros(N)
m = math.comb(n-t, t+1)
for h in range(N):
    noise = 0
    for i in range(math.comb(n, t+1)):
        noise += discretegauss.sample_dgauss(sigma2/m)
    noise_allsets[h] = noise
var_allsets = np.var(noise_allsets)
print("MSE is "+str(var_allsets))

print("partition:")
noise_partition = np.zeros(N)
ell = int(math.lcm(n,t+1)/(t+1))
m = math.ceil(ell*(1-t*(t+1)/n))
for h in range(N):
    noise = 0
    for i in range(ell):
        noise += discretegauss.sample_dgauss(sigma2/m)
    noise_partition[h] = noise
var_partition = np.var(noise_partition)
print("MSE is "+str(var_partition))


