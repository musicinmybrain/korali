import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.fft as fft
#import scipy.integrate as integrate
from scipy.integrate import simps
#from scipy import integrate


#################################### Plot Angular velocities

# pwd_orig = "/scratch/snx3000/anoca/CUP2D/longmpi/"

# # plot the angular velocity of the windmill
# data_vel = np.genfromtxt(pwd_orig + "velocity_0.dat", delimiter=' ', skip_header=1)
# data_vel_2 = np.genfromtxt(pwd_orig + "velocity_1.dat", delimiter=' ', skip_header=1)

# t_vel = data_vel[:, 0]
# ang_vel = data_vel[:, 9]
# ang_vel_2 = data_vel_2[:, 9]

# fig_angvel = plt.figure()
# plt.plot(t_vel, np.abs(np.abs(ang_vel) - np.abs(ang_vel_2)))
# #plt.plot(t_vel, ang_vel)
# fig_angvel.savefig('png/longmpi_angvel_diff.png')


# find the weird spike

# index = np.argmax(np.abs(ang_vel))
# index = 200

# fig_peak = plt.figure()
# plt.plot(t_vel[index:index + 30], ang_vel[index:index + 30])
# fig_peak.savefig('png/100smpi_exact__zoom_begin.png')

#################################### Plot torque applied

# pwd_orig = "/scratch/snx3000/anoca/CUP2D/100smpi_4nodes/"

# # plot the angular velocity of the windmill
# data_vel = np.genfromtxt(pwd_orig + "values_1.dat", delimiter=' ', skip_header=1)
# t_vel = data_vel[:, 0]
# torque = data_vel[:, 1]

# fig_angvel = plt.figure()
# plt.plot(t_vel, torque)
# fig_angvel.savefig('png/100smpi_4nodes_torque.png')

#################################### Plot the average profile (color map)

def average_profile(data, start, end):

    frequency = 0.5
    period = 1/frequency
    steps_per_period = 40
    num_periods = int(np.floor((end - start + 1) / steps_per_period))

    t = data[:, 0]

    avg_profs = np.zeros((num_periods, 32))

    for i in range(1, num_periods+1):
        index = start + i * steps_per_period
        T = t[start] + i * period
        avg_profs[i-1, :] = (1/(T - t[start])) * simps(data[start:index + 1, 1:], t[start:index + 1], axis = 0)

    return avg_profs


pwd_orig = "/scratch/snx3000/anoca/CUP2D/longmpi/"

data_prof = np.genfromtxt(pwd_orig + "velocity_profile_0.dat", delimiter=' ')
print(data_prof.shape)

fig_avg_color = plt.figure()

avg_ = average_profile(data_prof, 198, 7298)
print(avg_.shape)
plt.imshow(avg_, aspect='auto')
plt.xlabel('pos')
plt.ylabel('periods')
plt.colorbar()
plt.savefig('png/longmpi_avg.png')

# animation

fig_anim = plt.figure()

# low is at 0.35, high is at 1.05
# 0.7/32 = 0.021875 = height of each interval
height = 0.021875
x = np.linspace(0.35 + height/2, 1.05 - height/2, 32)

profile_orig = plt.plot(x, avg_[0, :], label='avg_profile')[0]
plt.plot(x, avg_[-1, :], label='avg_profile_end')
plt.xlabel('xpos')
plt.ylabel('|v|')
plt.ylim(-0.01,0.25)
plt.show()
plt.legend()



# function takes frame as an input
def AnimationFunction(frame):
    # line is set with new values of x and y
    profile_orig.set_data((x, avg_[frame, :]))
    #profile_orig.set_data((x, time_average))

anim_created = FuncAnimation(fig_anim, AnimationFunction, frames=177, interval=50, save_count=177)

anim_created.save('gif/longmpi.gif')

#################################### Plot the average profile mse curve

# def mse(data, start, end):
#     frequency = 0.5
#     period = 1/frequency
#     steps_per_period = 20
#     num_periods = int(np.floor((end - start + 1) / steps_per_period))

#     t = data[:, 0]

#     avg_profs = np.zeros((num_periods, 32))

#     for i in range(1, num_periods+1):
#         index = start + i * steps_per_period
#         T = t[start] + i * period
#         avg_profs[i-1, :] = (1/(T - t[start])) * simps(data[start:index + 1, 1:], t[start:index + 1], axis = 0)


#     diff = avg_profs[-1, :] - avg_profs[:, :]

#     mse = np.sqrt(np.sum(diff * diff, axis = 1))

#     return mse

# pwd_orig = "/scratch/snx3000/anoca/CUP2D/longmpi/"

# data_prof = np.genfromtxt(pwd_orig + "velocity_profile_0.dat", delimiter=' ')
# print(data_prof.shape)

# fig_mse = plt.figure()

# mse_long = mse(data_prof, 198, 7298)

# plt.plot(mse_long, 'og')

# plt.savefig('png/mse_longmpi.png')


#################################### Plot the average profile mse curve



# pwd_orig = "/scratch/snx3000/anoca/CUP2D/100s005/"

# pwd_orig_1 = "/scratch/snx3000/anoca/CUP2D/100s005_1/"

# pwd_orig_05 = "/scratch/snx3000/anoca/CUP2D/100s005_05/"

# pwd_orig_15 = "/scratch/snx3000/anoca/CUP2D/100s005_15/"

# pwd_orig_long = "/scratch/snx3000/anoca/CUP2D/100s005_long/"

# # pwd_2 = "/scratch/snx3000/anoca/CUP2D/100s005_1/"

# file_name = "velocity_profile_0.dat"

# data_orig = np.genfromtxt(pwd_orig + file_name, delimiter=' ')
# data_orig_1 = np.genfromtxt(pwd_orig_1 + file_name, delimiter=' ')
# data_orig_05 = np.genfromtxt(pwd_orig_05 + file_name, delimiter=' ')
# data_orig_15 = np.genfromtxt(pwd_orig_15 + file_name, delimiter=' ')

# data_orig_long = np.genfromtxt(pwd_orig_long + file_name, delimiter=' ')
# # data_2 = np.genfromtxt(pwd_2 + file_name, delimiter=' ')

# print(data_orig_long.shape)

# def average_profile(data, start, end):

#     frequency = 0.5
#     period = 1/frequency
#     steps_per_period = 40
#     num_periods = int(np.floor((end - start + 1) / steps_per_period))

#     t = data[:, 0]

#     avg_profs = np.zeros((num_periods, 32))

#     for i in range(1, num_periods+1):
#         index = start + i * steps_per_period
#         T = t[start] + i * period
#         avg_profs[i-1, :] = (1/(T - t[start])) * simps(data[start:index + 1, 1:], t[start:index + 1], axis = 0)

#     return avg_profs

# fig_avg_color = plt.figure()

# avg_ = average_profile(data_orig_long, 199, 15899)
# plt.imshow(avg_, aspect='auto')
# plt.xlabel('pos')
# plt.ylabel('periods')
# plt.colorbar()
# plt.savefig('png/avg_long.png')


"""
fig_col_avg, axs = plt.subplots(1, 4, figsize=(20, 5))

avg_ = average_profile(data_orig, 199, 1999)
avg_05 = average_profile(data_orig_05, 199, 1999)
avg_1 = average_profile(data_orig_1, 199, 1999)
avg_15 = average_profile(data_orig_15, 199, 1999)

axs[0].title.set_text('angle=0')
im = axs[0].imshow(avg_, aspect='auto')
plt.colorbar(im, ax=axs[0])
im.set_clim([0, 0.25])

axs[1].title.set_text('angle=0.5')
im = axs[1].imshow(avg_05, aspect='auto')
plt.colorbar(im, ax=axs[1])
im.set_clim([0, 0.25])

axs[2].title.set_text('angle=1')
im = axs[2].imshow(avg_1, aspect='auto')
plt.colorbar(im, ax=axs[2])
im.set_clim([0, 0.25])

axs[3].title.set_text('angle=1.5')
im = axs[3].imshow(avg_15, aspect='auto')
plt.colorbar(im, ax=axs[3])
im.set_clim([0, 0.25])

fig_col_avg.savefig('png/avg_comp.png')
"""

# fig_col = plt.figure()
# plt.imshow(data_orig_05[:, 1:], aspect='auto')
# plt.colorbar()
# plt.xlabel('profile')
# plt.ylabel('step')
# plt.title('100s005_05')
# plt.savefig('png/colormap_100s005_05.png')


# def mse(data, start, end):
#     frequency = 0.5
#     period = 1/frequency
#     steps_per_period = 40
#     num_periods = int(np.floor((end - start + 1) / steps_per_period))

#     t = data[:, 0]

#     avg_profs = np.zeros((num_periods, 32))

#     for i in range(1, num_periods+1):
#         index = start + i * steps_per_period
#         T = t[start] + i * period
#         avg_profs[i-1, :] = (1/(T - t[start])) * simps(data[start:index + 1, 1:], t[start:index + 1], axis = 0)


#     diff = avg_profs[-1, :] - avg_profs[:, :]

#     mse = np.sqrt(np.sum(diff * diff, axis = 1))

#     return mse


# fig_mse = plt.figure()

# mse_long = mse(data_orig_long, 199, 15899)

# plt.plot(mse_long, 'og')

# plt.savefig('png/mse_long.png')


# data_vel = np.genfromtxt(pwd_orig_long + "velocity_1.dat", delimiter=' ', skip_header=1)
# t_vel = data_vel[:, 0]
# ang_vel = data_vel[:, 9]


# f3 = plt.figure()
# plt.plot(t_vel, ang_vel)
# plt.xlabel('t')
# plt.ylabel('angvel')
# plt.savefig('angvel/100s005_long.png')

"""
mse100s005 = mse(data_orig)
mse100s005_05 = mse(data_orig_05)
mse100s005_1 = mse(data_orig_1)
mse100s005_15 = mse(data_orig_15)



fig = plt.figure()
ts = np.linspace(12, 100, 45)
plt.plot(ts, mse100s005, 'xk', label='angle=0')
plt.plot(ts, mse100s005_05, 'og', label='angle=0.5')
plt.plot(ts, mse100s005_1, '+b', label='angle=1')
plt.plot(ts, mse100s005_15, '.r', label='angle=15')
plt.xlabel('time')
plt.ylabel('mse')
plt.legend()

plt.savefig('png/mse_comp.png')

"""

# fig_prof = plt.figure()

# t = data_orig[start:, 0]

# integ = (1/75) * simps(data_orig[start:, 1:], t, axis = 0)

# std_ = data_orig[start:, 1:].std(axis=0)
# print(std_)

# plt.plot(integ)
# plt.plot(integ + std_)
# plt.plot(integ - std_)

# plt.savefig('png/avg_prof_100s005.png')


# fig_prof = plt.figure()

# sp = fft.fftshift(fft.fft(integ))
# freq = fft.fftshift(fft.fftfreq(32))
# plt.plot(freq, sp.real, label='real')
# plt.plot(freq, sp.imag, label='imag')
# plt.xlabel('omega')
# plt.legend()

# plt.savefig('fourier/avg_prof_100s005.png')



"""
f = plt.figure()

# low is at 0.35, high is at 1.05
# 0.7/32 = 0.021875 = height of each interval
height = 0.021875
x = np.linspace(0.35 + height/2, 1.05 - height/2, 32)

profile_orig = plt.plot(x, data_orig[0, 1:], label='orig_profile')[0]
avg = np.mean(data_orig[:, 1:], axis=0)
plt.plot(x, avg, label='avg')
plt.xlabel('xpos')
plt.ylabel('|v|')
plt.ylim(-0.01,0.45)
plt.show()
plt.legend()

#plt.savefig('png/freqsquare2hz.png')
"""

# f1 = plt.figure()

# t_orig = data_orig[:, 0]

# t2 = data_2[:, 0]

# #diff = abs(data_orig[:, 1:] - data_2[:, 1:])

# # get cumulative average of the values over the time of the simulation
# orig = np.cumsum(data_orig[:, 16])
# orig_2 = np.cumsum(data_2[:, 16])

# #plt.plot(t_orig, np.abs(data_orig[:, 16] - data_2[:, 16]))
# plt.plot(t_orig, np.abs(orig - orig_2) / np.abs(orig))

# plt.xlabel('t')
# plt.ylabel('diff')

# plt.savefig('diffcum.png')



"""

# plot the average of one element over the time of the simulation
f2 = plt.figure()

t = data_orig[:,0]

# diff = data_orig[1:, 16] - data_orig[:-1, 16]

# plt.plot(t, data_orig[:, 16])
plt.plot(t, data_orig[:, 16])
plt.xlabel('t')
plt.ylabel('|v|')
#plt.ylim(-0.01,0.45)
plt.savefig('png/100s005.png')

f3 = plt.figure()

sp = fft.fftshift(fft.fft(data_orig[:, 16]))
freq = fft.fftshift(fft.fftfreq(t.shape[-1]))
plt.plot(freq, sp.real, label='real')
plt.plot(freq, sp.imag, label='imag')
plt.xlim(-0.1, 0.1)
plt.xlabel('omega')
plt.legend()

plt.savefig('fourier/100s005_1.png')


# plot the angular velocity of the windmill
data_vel = np.genfromtxt(pwd_orig + "velocity_1.dat", delimiter=' ', skip_header=1)
t_vel = data_vel[:, 0]
ang_vel = data_vel[:, 9]


# f3 = plt.figure()
# plt.plot(t_vel, ang_vel)
# plt.xlabel('t')
# plt.ylabel('angvel')
# plt.savefig('angvel/100s005.png')


"""








"""
# function takes frame as an input
def AnimationFunction(frame):
    # line is set with new values of x and y
    profile_orig.set_data((x, data_orig[10*frame, 1:]))
    #profile_orig.set_data((x, time_average))

anim_created = FuncAnimation(f, AnimationFunction, frames=300, interval=100, save_count=300)

anim_created.save('gif/torque1e_4.gif')
"""

""""""

# compute the reward obtained for this particular case

"""pwd = "/scratch/snx3000/anoca/CUP2D/forced_rot_long40/"
file_name = "velocity_profile_0.dat"
data = np.genfromtxt(pwd + file_name, delimiter=' ')

profiles = data[:, 1:]
diff = true_prof - profiles
square_diff = diff * diff
D = 10
sum_square_diff = -D * np.sum(np.sum(square_diff))
print(sum_square_diff)"""