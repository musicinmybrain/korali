# computes the results for each test and stores it into the _results_test/ folder
import numpy as np
# ---------- Accessing files ----------

path_to_test = '_results_windmill_testing/'

folders_test = ['both', 'energy_zero', 'flow_zero_4', 'flow_zero_3', 'uncontrolled']

output = '_results_test/'



def genResults(foldername):
  # force and velocity values go from 0 to num_windmills - 1
  num_windmills = 2

  values_name = 'values_'
  vels_name = 'targetvelocity_'
  rew_name = 'rewards_'

  # contains link to all the sample folders in the test results
  num_trials = 64
  sample = 'sample000000'
  folders = [path_to_test + foldername + "/" + sample + "{:02d}".format(trial) + '/' for trial in range(num_trials)]

  # shape num_trials x num_windmills
  # file names
  value_files = [ [folder_ + values_name + str(num) + '.dat' for num in range(num_windmills)] for folder_ in folders]

  vel_files = [ [folder_ + vels_name + str(num) + '.dat' for num in range(num_windmills)] for folder_ in folders]

  rew_files = [ [folder_ + rew_name + str(num) + '.dat' for num in range(num_windmills)] for folder_ in folders]


  dat = np.genfromtxt(value_files[0][0], delimiter=' ')

  num_steps = dat.shape[0]
  print(num_steps)

  num_el = dat.shape[1]


  # arrays
  values = np.zeros((num_trials,  num_windmills, num_steps, num_el))

  vels = np.zeros((num_trials, num_steps,  2))

  rewards = np.zeros((num_trials, num_windmills, num_steps, 3))

  # factor to non dimensionalize the torque
  u = 0.15
  a = 0.0405



  for trial in range(num_trials):
    for mill in range(num_windmills):
      data = np.genfromtxt(value_files[trial][mill], delimiter=' ')

      # nondimensionalize the time
      data[:, 0] *= u / a

      # nondimensionalize the torque
      data[:, 1] /= (u**2 * a**2)

      # nondimensionalize the angular velocity
      data[:, 3] *= a / u

      print(data.shape)
      values[trial, mill, :, : ] = data

      reward = np.genfromtxt(rew_files[trial][mill], delimiter=' ')
      # nondimensionalize the time
      reward[:, 0] *= u / a
      rewards[trial, mill, :, :] = reward


    velo = np.genfromtxt(vel_files[trial][0], delimiter=' ')
    # nondimensionalize the time
    velo[:, 0] *= u / a
    vels[trial, :, :] = velo

  # compute the interesting values

  # time, action, state (std and mean) for both windmills

  # fan 1
  tau_mean_0 = np.mean(values[:,0,:, 1], axis=0)
  tau_std_0 = np.std(values[:,0,:, 1], axis=0)
  ang_mean_0 = np.mean(values[:,0, :, 2], axis=0)
  ang_std_0 = np.std(values[:,0, :, 2], axis=0)
  ang_vel_mean_0 = np.mean(values[:,0, :, 3], axis=0)
  ang_vel_std_0 = np.std(values[:,0, :, 3], axis=0)


  # fan 2
  tau_mean_1 = np.mean(values[:,1,:, 1], axis=0)
  tau_std_1 = np.std(values[:,1,:, 1], axis=0)
  ang_mean_1 = np.mean(values[:,1, :, 2], axis=0)
  ang_std_1 = np.std(values[:,1, :, 2], axis=0)
  ang_vel_mean_1 = np.mean(values[:,1, :, 3], axis=0)
  ang_vel_std_1 = np.std(values[:,1, :, 3], axis=0)

  # first element is time
  out = np.stack( (values[0, 0, :, 0], tau_mean_0, tau_std_0, tau_mean_1, tau_std_1, 
                                            ang_mean_0, ang_std_0, ang_mean_1, ang_std_1,
                                            ang_vel_mean_0, ang_vel_std_0, ang_vel_mean_1, ang_vel_std_1), axis=1)
  np.save(output + foldername + "_values.npy", out)


  # velocity at target point vs time (mean and std)

  vels_mean = np.mean(vels, axis=0)
  
  vels_std = np.std(vels, axis=0)

  out2 = np.stack( (vels_mean[:, 0], vels_mean[:, 1], vels_std[:, 1]), axis=1)
  np.save(output + foldername + "_vels.npy", out2)


  # rewards vs time for the two fans
  en_mean_0 = np.mean(rewards[:,0,:, 1], axis=0)
  en_std_0 = np.std(rewards[:,0,:, 1], axis=0)
  flow_mean_0 = np.mean(rewards[:,0, :, 2], axis=0)
  flow_std_0 = np.std(rewards[:,0, :, 2], axis=0)

  en_mean_1 = np.mean(rewards[:,1,:, 1], axis=0)
  en_std_1 = np.std(rewards[:,1,:, 1], axis=0)
  flow_mean_1 = np.mean(rewards[:,1, :, 2], axis=0)
  flow_std_1 = np.std(rewards[:,1, :, 2], axis=0)

  out3 = np.stack( (rewards[0, 0, :, 0], en_mean_0, en_std_0, en_mean_1, en_std_1, 
                                        flow_mean_0, flow_std_0, flow_mean_1, flow_std_1), axis=1)

  np.save(output + foldername + "_rews.npy", out3)



for ind, folder in enumerate(folders_test):
  genResults(folder)


