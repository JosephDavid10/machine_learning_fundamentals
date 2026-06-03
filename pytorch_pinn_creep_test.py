import torch
import numpy as np
import matplotlib.pyplot as plt
#================================================================
# CREEP TEST SIMULATION FOR GENERALIZED MAXWELL MODEL
# Numerical method: Explicit Euler
#================================================================
# creating data to define material's properties
E_inf = torch.tensor(10)  # stiffness when time approaches infinity
E_i = torch.tensor([20, 10, 15])  # the stiffness of the spring in each branch
eta_i = torch.tensor([200, 150, 300])  # damping coefficient of the dashpots in each branch
E_0 = E_inf + torch.sum(E_i)  # total stiffness of the system

# Constant stress
sigma_0 = torch.tensor(10)

# time steps
dt, total_time = 0.05, 250
step = int(total_time/dt)
t = torch.linspace(0, total_time, step)

epsilon_list = []
epsilon_i = sigma_0/E_0  # the instant strain
epsilon_list.append(epsilon_i)
sigma_i = E_i * epsilon_i  # Initial total strain in each branch

for i in range(1, step):
    # differential equation: E*deps/dt = dsigma/dt + (E/eta)*sigma
    damp = torch.sum(sigma_i*(E_i / eta_i))
    deps_dt = damp/E_0
  
    # Explicit euler e(k+1) = e(k) + de/dt * dt
    epsilon_i = epsilon_i + (deps_dt)*dt
    epsilon_list.append(epsilon_i)

    # to calculate the new stress value in each branch
    ds_dt = E_i * deps_dt - (sigma_i / (eta_i / E_i))
    sigma_i = sigma_i + ds_dt * dt

epsilon_real = torch.stack(epsilon_list)

plt.plot(t, epsilon_real)
