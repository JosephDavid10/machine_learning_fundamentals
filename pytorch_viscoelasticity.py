# creep test for generalized maxwell model
# creating data with analytical formulation
E_inf = torch.tensor(50)
E_i = torch.tensor(50)
eta_i = torch.tensor(20)
E_0 = E_inf + sum(E_i)

sigma_0 = torch.tensor(10)

# time steps
dt, total_time = 0.05, 250
step = total_time/dt
t = torch.linspace(0, total_time, step)

epsilon_list = []
epsilon_i = sigma_0/E_0
epsilon_list.append(epsilon_i)
sigma_i = E_i*epsilon_i

for i in range(1, step):

    soma_relax = torch.sum(sigma_i*E_i / eta_i)
    deps_dt = soma_relax/E_0
    epsilon_now = epsilon_i + (deps_dt)*dt
    epsilon_list.append(epsilon_i)
    
    ds_dt = E_i * deps_dt - (sigma_i / (eta_i / E_i))
    sigma_i_real = sigma_i + ds_dt * dt
#print(epsilon_list)
epsilon_real = torch.stack(epsilon_list)

plt.plot(t, epsilon_real)
