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


#===========================================
# INVERSION MODEL
#===========================================
# estimating initial inputs
E_inf_est = torch.tensor(25., requires_grad = True)  
E_i_est = torch.tensor([25., 5., 20.], requires_grad = True)  
tau_i_est = torch.tensor([10., 15., 20.], requires_grad = False) 

optimizer = torch.optim.Adam([E_inf_est, E_i_est], lr = 0.1) 

# parameter estimation
epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()

    E_0_est = E_inf_est + torch.sum(E_i_est)
    epsilon_i_est = sigma_0/E_0_est
    sigma_i_est = E_i_est * epsilon_i_est

    epsilon_est = torch.zeros(step)
    epsilon_est[0] = epsilon_i_est 
    deps_dt_est = 0.0
    for k in range(1, step):
        deps_dt_est =  torch.sum(sigma_i_est*(1/tau_i_est))/E_0_est
        epsilon_i_est = epsilon_i_est + deps_dt_est*dt
        epsilon_est[k] = epsilon_i_est
        ds_dt_est = E_i_est * deps_dt_est - (sigma_i_est / tau_i_est)
        sigma_i_est = sigma_i_est + ds_dt_est * dt

    E_i_formatado = [f"{x:.2f}" for x in E_i_est.tolist()]
        
    # loss function
    loss = torch.mean((epsilon_real - epsilon_est)**2)

    loss.backward()
    optimizer.step()
    
     # Exibe o progresso a cada 50 iterações
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Época {epoch:03d} | Erro: {loss.item():.6f} | E_inf: {E_inf_est.item():.2f} | E_i: {E_i_formatado}")
