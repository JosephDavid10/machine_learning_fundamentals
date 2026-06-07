import torch
import numpy as np
import matplotlib.pyplot as plt
#================================================================
# STRESS RELAXATION TEST SIMULATION FOR GENERALIZED MAXWELL MODEL
# Numerical method: Explicit Euler
#================================================================
# creating data to define material's properties
E_inf = torch.tensor(10)
E_i = torch.tensor([20, 10, 15])
eta_i = torch.tensor([200, 150, 300])

epsilon_0 = torch.tensor(0.2)

sigma_inf, sigma_i = E_inf*epsilon_0, E_i*epsilon_0

# time steps
dt, total_time = 0.05, 250
step = int(total_time/dt)
t = torch.linspace(0, total_time, step)

sigma_list = []
sigma_0 = torch.sum(sigma_i) + sigma_inf
sigma_list.append(sigma_0)

for i in range(1, step):

    ds_dt = - (sigma_i / (eta_i / E_i))
    sigma_i = sigma_i + ds_dt*dt
    sigma_0 = torch.sum(sigma_i) + sigma_inf
    sigma_list.append(sigma_0)

sigma_real = torch.stack(sigma_list)

plt.plot(t, sigma_real)
