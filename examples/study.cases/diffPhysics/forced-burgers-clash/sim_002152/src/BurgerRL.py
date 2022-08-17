# This script is a duplicate of the script on this site:
# https://www.physicsbaseddeeplearning.org/reinflearn-code.html#training-via-reinforcement-learning

# Use those versions and githubs
# !pip install stable-baselines3==1.1 phiflow==1.5.1
# !git clone https://github.com/Sh0cktr4p/PDE-Control-RL.git
# !git clone https://github.com/holl-/PDE-Control.git

import sys; sys.path.append('PDE-Control/src'); sys.path.append('PDE-Control-RL/src')
import time, csv, os, shutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from phi.flow import *
import burgers_plots as bplt
import matplotlib.pyplot as plt
from envs.burgers_util import GaussianClash, GaussianForce

# DATA GENERATION

DOMAIN = Domain([32], box=box[0:1])     # Size and shape of the fields
VISCOSITY = 0.003
STEP_COUNT = 32                         # Trajectory length
DT = 0.03
DIFFUSION_SUBSTEPS = 1

DATA_PATH = 'forced-burgers-clash'
SCENE_COUNT = 1000
BATCH_SIZE = 100

TRAIN_RANGE = range(200, 1000)
VAL_RANGE = range(100, 200)
TEST_RANGE = range(0, 100)

for batch_index in range(SCENE_COUNT // BATCH_SIZE):
    scene = Scene.create(DATA_PATH, count=BATCH_SIZE)
    print(scene)
    world = World()
    u0 = BurgersVelocity(
        DOMAIN, 
        velocity=GaussianClash(BATCH_SIZE), 
        viscosity=VISCOSITY, 
        batch_size=BATCH_SIZE, 
        name='burgers'
    )
    u = world.add(u0, physics=Burgers(diffusion_substeps=DIFFUSION_SUBSTEPS))
    force = world.add(FieldEffect(GaussianForce(BATCH_SIZE), ['velocity']))
    scene.write(world.state, frame=0)
    for frame in range(1, STEP_COUNT + 1):
        world.step(dt=DT)
        scene.write(world.state, frame=frame)

# TRAINING VIA REINFORCEMENT LEARNING

from experiment import BurgersTraining

N_ENVS = 10                         # On how many environments to train in parallel, load balancing
FINAL_REWARD_FACTOR = STEP_COUNT    # Penalty for not reaching the goal state
STEPS_PER_ROLLOUT = STEP_COUNT * 10 # How many steps to collect per environment between agent updates
N_EPOCHS = 10                       # How many epochs to perform during each agent update
RL_LEARNING_RATE = 1e-4             # Learning rate for agent updates
RL_BATCH_SIZE = 128                 # Batch size for agent updates
RL_ROLLOUTS = 500                  # Number of iterations for RL training

rl_trainer = BurgersTraining(
    path='PDE-Control-RL/networks/rl-models/bench', # Replace path to train a new model
    domain=DOMAIN,
    viscosity=VISCOSITY,
    step_count=STEP_COUNT,
    dt=DT,
    diffusion_substeps=DIFFUSION_SUBSTEPS,
    n_envs=N_ENVS,
    final_reward_factor=FINAL_REWARD_FACTOR,
    steps_per_rollout=STEPS_PER_ROLLOUT,
    n_epochs=N_EPOCHS,
    learning_rate=RL_LEARNING_RATE,
    batch_size=RL_BATCH_SIZE,
    data_path=DATA_PATH,
    val_range=VAL_RANGE,
    test_range=TEST_RANGE,
)

# Optional code for debugging
# %load_ext tensorboard
# %tensorboard --logdir PDE-Control-RL/networks/rl-models/bench/tensorboard-log

rl_trainer.train(n_rollouts=RL_ROLLOUTS, save_freq=50)

# RL EVALUATION

TEST_SAMPLE = 0    # Change this to display a reconstruction of another scene
rl_frames, gt_frames, unc_frames = rl_trainer.infer_test_set_frames()

fig, axs = plt.subplots(1, 3, figsize=(18.9, 9.6))
axs[0].set_title("Reinforcement Learning"); axs[1].set_title("Ground Truth"); axs[2].set_title("Uncontrolled")
for plot in axs:
    plot.set_ylim(-2, 2); plot.set_xlabel('x'); plot.set_ylabel('u(x)')

for frame in range(0, STEP_COUNT + 1):
    frame_color = bplt.gradient_color(frame, STEP_COUNT+1);
    axs[0].plot(rl_frames[frame][TEST_SAMPLE,:], color=frame_color, linewidth=0.8)
    axs[1].plot(gt_frames[frame][TEST_SAMPLE,:], color=frame_color, linewidth=0.8)
    axs[2].plot(unc_frames[frame][TEST_SAMPLE,:], color=frame_color, linewidth=0.8)

# DIFFERENTIABLE PHYISICS TRAINING

from control.pde.burgers import BurgersPDE
from control.control_training import ControlTraining
from control.sequences import StaggeredSequence

dp_app = ControlTraining(
    STEP_COUNT,
    BurgersPDE(DOMAIN, VISCOSITY, DT),
    datapath=DATA_PATH,
    val_range=VAL_RANGE,
    train_range=TRAIN_RANGE,
    trace_to_channel=lambda trace: 'burgers_velocity',
    obs_loss_frames=[],
    trainable_networks=['CFE'],
    sequence_class=StaggeredSequence,
    batch_size=100,
    view_size=20,
    learning_rate=1e-3,
    learning_rate_half_life=1000,
    dt=DT
).prepare()

DP_TRAINING_ITERATIONS = 10000  # Change this to change training duration

dp_training_eval_data = []
start_time = time.time()

for epoch in range(DP_TRAINING_ITERATIONS):
    dp_app.progress()
    # Evaluate validation set at regular intervals to track learning progress
    # Size of intervals determined by RL epoch count per iteration for accurate comparison
    if epoch % N_EPOCHS == 0:
        f = dp_app.infer_scalars(VAL_RANGE)['Total Force'] / DT
        dp_training_eval_data.append((time.time() - start_time, epoch, f))

DP_STORE_PATH = 'networks/dp-models/bench'
if not os.path.exists(DP_STORE_PATH):
    os.makedirs(DP_STORE_PATH)

# store training progress information
with open(os.path.join(DP_STORE_PATH, 'val_forces.csv'), 'at') as log_file:
    logger = csv.DictWriter(log_file, ('time', 'epoch', 'forces'))
    logger.writeheader()
    for (t, e, f) in dp_training_eval_data:
        logger.writerow({'time': t, 'epoch': e, 'forces': f})

dp_checkpoint = dp_app.save_model()
shutil.move(dp_checkpoint, DP_STORE_PATH)

# dp_path = 'PDE-Control-RL/networks/dp-models/bench/checkpoint_00020000/'
# networks_to_load = ['OP2', 'OP4', 'OP8', 'OP16', 'OP32']

# dp_app.load_checkpoints({net: dp_path for net in networks_to_load})

dp_frames = dp_app.infer_all_frames(TEST_RANGE)
dp_frames = [s.burgers.velocity.data for s in dp_frames]
_, gt_frames, unc_frames = rl_trainer.infer_test_set_frames()

TEST_SAMPLE = 0    # Change this to display a reconstruction of another scene
fig, axs = plt.subplots(1, 3, figsize=(18.9, 9.6))

axs[0].set_title("Differentiable Physics")
axs[1].set_title("Ground Truth")
axs[2].set_title("Uncontrolled")

for plot in axs:
    plot.set_ylim(-2, 2)
    plot.set_xlabel('x')
    plot.set_ylabel('u(x)')

for frame in range(0, STEP_COUNT + 1):
    frame_color = bplt.gradient_color(frame, STEP_COUNT+1)
    axs[0].plot(dp_frames[frame][TEST_SAMPLE,:], color=frame_color, linewidth=0.8)
    axs[1].plot(gt_frames[frame][TEST_SAMPLE,:], color=frame_color, linewidth=0.8)
    axs[2].plot(unc_frames[frame][TEST_SAMPLE,:], color=frame_color, linewidth=0.8)


