import matplotlib.pyplot as plt
import numpy as np

path_to_folders = '_results_test/'

# folders_test = ['both', 'energy_zero', 'flow_zero_4', 'flow_zero_3', 'uncontrolled']

folders_test = ['both', 'energy_zero', 'flow_zero_4', 'uncontrolled',  'flow_zero_3']

names = ['both', 'only energy', 'only flow: tau 1e-4', 'uncontrolled', 'only flow: tau 1e-3', 'dummy']

path_to_files = [path_to_folders + folder for folder in folders_test]

output = '_results_plots/'

dico_values = {}
dico_vels = {}
dico_rewards = {}


### load data
###############################################################
for index, path in enumerate(path_to_files):
  dico_values[names[index]] = np.load(path + "_values.npy")
  dico_vels[names[index]] = np.load(path + "_vels.npy")
  dico_rewards[names[index]] = np.load(path + "_rews.npy")
###############################################################


### PLOTS


colors = ['blue', 'red', 'green', 'black', 'purple']

# dark light for each color
colors_both = ['darkblue', 'cyan', 'darkred', 'salmon', 'darkgreen', 'limegreen', 'dimgray', 'lightgray', 'darkviolet', 'violet']

###############################################################
################# FANS separate
# ------- compare both, energy, flow, uncontrolled

# ) torque vs time for fan 1
fig1 = plt.figure(1)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 0], dico_values[name][:, 1], colors[ind], label=name)
  plt.fill_between(dico_values[name][:, 0], dico_values[name][:, 1] - dico_values[name][:, 2], dico_values[name][:, 1] + dico_values[name][:, 2], color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$C_\tau$ []')
plt.legend()
plt.title("Torque : Fan 1")
plt.grid()

fig1.set_size_inches(5, 5)
fig1.tight_layout()
fig1.savefig(output + "tau_fan_1.png")

# ) torque vs time for fan 2
fig2 = plt.figure(2)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 0], dico_values[name][:, 3], colors[ind], label=name)
  plt.fill_between(dico_values[name][:, 0], dico_values[name][:, 3] - dico_values[name][:, 4], dico_values[name][:, 3] + dico_values[name][:, 4], color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$C_\tau$ []')
plt.legend()
plt.title("Torque : Fan 2")
plt.grid()

fig2.set_size_inches(5, 5)
fig2.tight_layout()
fig2.savefig(output + "tau_fan_2.png")


# ) omega vs time for fan 1
fig3 = plt.figure(3)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 0], dico_values[name][:, 9], colors[ind], label=name)
  plt.fill_between(dico_values[name][:, 0], dico_values[name][:, 9] - dico_values[name][:, 10], dico_values[name][:, 9] + dico_values[name][:, 10], color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$C_\omega$ []')
plt.legend()
plt.title("Angular velocity : Fan 1")
plt.grid()

fig3.set_size_inches(5, 5)
fig3.tight_layout()
fig3.savefig(output + "omega_fan_1.png")

# ) omega vs time for fan 2
fig4 = plt.figure(4)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 0], dico_values[name][:, 11], colors[ind], label=name)
  plt.fill_between(dico_values[name][:, 0], dico_values[name][:, 11] - dico_values[name][:, 12], dico_values[name][:, 11] + dico_values[name][:, 12], color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$C_\omega$ []')
plt.legend()
plt.title("Angular velocity : Fan 2")
plt.grid()

fig4.set_size_inches(5, 5)
fig4.tight_layout()
fig4.savefig(output + "omega_fan_2.png")

# ) angle vs time for fan 1
figang1 = plt.figure(50)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 0], dico_values[name][:, 5], colors[ind], label=name)

plt.xlabel(r"$t$ []")
plt.ylabel(r'$\phi$ []')
plt.legend()
plt.title("Angle : Fan 1")
plt.grid()

figang1.set_size_inches(5, 5)
figang1.tight_layout()
figang1.savefig(output + "phi_fan_1.png")

# ) angle vs time for fan 2
figang2 = plt.figure(51)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 0], dico_values[name][:, 7], colors[ind], label=name)

plt.xlabel(r"$t$ []")
plt.ylabel(r'$\phi$ []')
plt.legend()
plt.title("Angle : Fan 2")
plt.grid()

figang2.set_size_inches(5, 5)
figang2.tight_layout()
figang2.savefig(output + "phi_fan_2.png")


# ) omega vs angle for fan 1
figphase1 = plt.figure(52)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 5], dico_values[name][:, 9], colors[ind], label=name)

plt.xlabel(r"$\phi$ []")
plt.ylabel(r'$C_\omega$ []')
plt.legend()
plt.title("Phase space : Fan 1")
plt.grid()

figphase1.set_size_inches(5, 5)
figphase1.tight_layout()
figphase1.savefig(output + "phase_fan_1.png")

# ) omega vs angle for fan 2
figphase2 = plt.figure(53)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  plt.plot(dico_values[name][:, 7], dico_values[name][:, 11], colors[ind], label=name)

plt.xlabel(r"$\phi$ []")
plt.ylabel(r'$C_\omega$ []')
plt.legend()
plt.title("Phase space : Fan 2")
plt.grid()


figphase2.set_size_inches(5, 5)
figphase2.tight_layout()
figphase2.savefig(output + "phase_fan_2.png")

# ) torque vs omega for both
fig5 = plt.figure(5)
name_both = 'both'

# all
plt.plot(dico_values[name_both][:, 9], dico_values[name_both][:, 1], colors_both[0], marker=',', label=name + " fan 1")
plt.plot(dico_values[name_both][:, 11], dico_values[name_both][:, 3], colors_both[1], marker=',', label= name + " fan 2")
# for the first 5 seconds

begin = 400
plt.plot(dico_values[name_both][:begin:10, 9], dico_values[name_both][:begin:10, 1], 'gx')
plt.plot(dico_values[name_both][:begin:10, 11], dico_values[name_both][:begin:10, 3], 'gx')

plt.xlabel(r"$C_\omega$ []")
plt.ylabel(r'$C_\tau$ []')
plt.legend(loc='best')
plt.title("Policy both (action vs state)")
plt.grid()

fig5.set_size_inches(5, 5)
fig5.tight_layout()
fig5.savefig(output + "policy_both.png")

# ) torque vs omega for flow 4
fig6 = plt.figure(6)

name_4 = 'only flow: tau 1e-4'


plt.plot(dico_values[name_4][:, 9], dico_values[name_4][:, 1], colors_both[4], marker=',', label=name + " fan 1")
plt.plot(dico_values[name_4][:, 11], dico_values[name_4][:, 3], colors_both[5], marker=',', label= name + " fan 2")

begin = 500
plt.plot(dico_values[name_4][:begin:10, 9], dico_values[name_4][:begin:10, 1], 'kx')
plt.plot(dico_values[name_4][:begin:10, 11], dico_values[name_4][:begin:10, 3], 'kx')


plt.xlabel(r"$C_\omega$ []")
plt.ylabel(r'$C_\tau$ []')
plt.legend(loc='best')
plt.title("Policy flow e-4 (action vs state)")
plt.grid()

fig6.set_size_inches(5, 5)
fig6.tight_layout()
fig6.savefig(output + "policy_e4.png")


# ) torque vs omega for flow 3
fig73 = plt.figure(73)

name_3 = 'only flow: tau 1e-3'


plt.plot(dico_values[name_3][:, 9], dico_values[name_3][:, 1], colors_both[-2], marker=',', label=name + " fan 1")
plt.plot(dico_values[name_3][:, 11], dico_values[name_3][:, 3], colors_both[-1], marker=',', label= name + " fan 2")

begin = 300
plt.plot(dico_values[name_3][:begin:10, 9], dico_values[name_3][:begin:10, 1], 'kx')
plt.plot(dico_values[name_3][:begin:10, 11], dico_values[name_3][:begin:10, 3], 'kx')


plt.xlabel(r"$C_\omega$ []")
plt.ylabel(r'$C_\tau$ []')
plt.legend(loc='best')
plt.title("Policy flow e-3 (action vs state)")
plt.grid()

fig73.set_size_inches(5, 5)
fig73.tight_layout()
fig73.savefig(output + "policy_e3.png")

####################################################
# plot all three of the the policy spaces in on subplot

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

name_both = 'both'

# all
ax1.plot(dico_values[name_both][:, 9], dico_values[name_both][:, 1], colors_both[0], marker=',', label=name_both + " fan 1")
ax1.plot(dico_values[name_both][:, 11], dico_values[name_both][:, 3], colors_both[1], marker=',', label= name_both + " fan 2")
# for the first 5 seconds

begin = 400
ax1.plot(dico_values[name_both][:begin:10, 9], dico_values[name_both][:begin:10, 1], 'gx')
ax1.plot(dico_values[name_both][:begin:10, 11], dico_values[name_both][:begin:10, 3], 'gx')

ax1.set_xlabel(r"$C_\omega$ []")
ax1.set_ylabel(r'$C_\tau$ []')
ax1.legend(loc=(0.4, 0.3))
ax1.grid()
ax1.set_title("both")


name_4 = 'only flow: tau 1e-4'

ax2.plot(dico_values[name_4][:, 9], dico_values[name_4][:, 1], colors_both[4], marker=',', label= "flow 1e-4 fan 1")
ax2.plot(dico_values[name_4][:, 11], dico_values[name_4][:, 3], colors_both[5], marker=',', label= "flow 1e-4 fan 2")

begin = 500
ax2.plot(dico_values[name_4][:begin:10, 9], dico_values[name_4][:begin:10, 1], 'kx')
ax2.plot(dico_values[name_4][:begin:10, 11], dico_values[name_4][:begin:10, 3], 'kx')

ax2.set_xlabel(r"$C_\omega$ []")
ax2.legend(loc='best')
ax2.grid()
ax2.set_title("flow e-4")


name_3 = 'only flow: tau 1e-3'

ax3.plot(dico_values[name_3][:, 9], dico_values[name_3][:, 1], colors_both[-2], marker=',', label= "flow 1e-3 fan 1")
ax3.plot(dico_values[name_3][:, 11], dico_values[name_3][:, 3], colors_both[-1], marker=',', label= "flow 1e-3 fan 2")

begin = 300
ax3.plot(dico_values[name_3][:begin:10, 9], dico_values[name_3][:begin:10, 1], 'kx')
ax3.plot(dico_values[name_3][:begin:10, 11], dico_values[name_3][:begin:10, 3], 'kx')

ax3.set_xlabel(r"$C_\omega$ []")
ax3.legend(loc='best')
ax3.grid()
ax3.set_title("flow e-3")


fig.set_size_inches(10, 5)
fig.tight_layout()
fig.savefig(output + "policies.png")

###################################################3


# ) reward vs time for both fans, only energy, summed up
fig7 = plt.figure(7)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  mean = dico_rewards[name][:, 1] + dico_rewards[name][:, 3]
  std = dico_rewards[name][:, 2] + dico_rewards[name][:, 4]
  plt.plot(dico_rewards[name][:, 0], mean, colors[ind], label=name)
  plt.fill_between(dico_rewards[name][:, 0], mean - std, mean + std, color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$r$ []')
plt.legend()
plt.title("Rewards : only energy")
plt.grid()

fig7.set_size_inches(5, 5)
fig7.tight_layout()
fig7.savefig(output + "rew_energy.png")

# ) reward vs time for both fans, only flow
fig8 = plt.figure(8)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  mean = dico_rewards[name][:, 5] + dico_rewards[name][:, 7]
  std = dico_rewards[name][:, 6] + dico_rewards[name][:, 8]
  plt.plot(dico_rewards[name][:, 0], mean, colors[ind], label=name)
  plt.fill_between(dico_rewards[name][:, 0], mean - std, mean + std, color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$r$ []')
plt.legend()
plt.title("Rewards : only flow")
plt.grid()

fig8.set_size_inches(5, 5)
fig8.tight_layout()
fig8.savefig(output + "rew_flow.png")


# ) reward vs time for both fans, all
fig9 = plt.figure(9)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  mean = dico_rewards[name][:, 1] + dico_rewards[name][:, 3] + dico_rewards[name][:, 5] + dico_rewards[name][:, 7]
  std = dico_rewards[name][:, 2] + dico_rewards[name][:, 4] + dico_rewards[name][:, 6] + dico_rewards[name][:, 8]
  plt.plot(dico_rewards[name][:, 0], mean, colors[ind], label=name)
  plt.fill_between(dico_rewards[name][:, 0], mean - std, mean + std, color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$r$ []')
plt.legend()
plt.title("Rewards : all")
plt.grid()

fig9.set_size_inches(5, 5)
fig9.tight_layout()
fig9.savefig(output + "rew_all.png")



# ) velocity at target vs time for 4 different runs, not using the std for uncontrolled and energy
fig10 = plt.figure(10)

for ind, name in enumerate(names[:-1]): # don't do for the last one
  mean = dico_vels[name][:, 1]
  std = dico_vels[name][:, 2]
  plt.plot(dico_vels[name][:, 0], mean, colors[ind], label=name)
  if (name != "uncontrolled") and (name != "only energy"):
    plt.fill_between(dico_vels[name][:, 0], mean - std, mean + std, color=colors[ind], alpha=0.2)

plt.xlabel(r"$C_t$ []")
plt.ylabel(r'$v_{target}$ []')
plt.legend()
plt.title("Velocity at target")
plt.grid()

fig10.set_size_inches(10, 5)
fig10.tight_layout()
fig10.savefig(output + "velocity.png")



plt.show(block=False)

plt.pause(1) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run

###############################################################
