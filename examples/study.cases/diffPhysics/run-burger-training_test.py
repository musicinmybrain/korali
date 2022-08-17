# NOTE: THIS IS JUST A VERY FIRST VERSION OF THE SCRIPT
# This version will be used as a baseline to create a much shorter and simpler version
# It is left in the project folder as a backup
# This version requires that the directory forced-burgers-clash is in the same directory than this file

# Use the following versions of PhiFlow and stable-baselines3
# !pip install stable-baselines3==1.1 phiflow==1.5.1

import time, csv, os, shutil
from phi.flow import *
from phi.tf.flow import *

#from control.pde.burgers import BurgersPDE
#from control.control_training import ControlTraining
#from control.sequences import StaggeredSequence

# Variables needed in classes below
TYPE_UNKNOWN = 0
TYPE_PLANNED = 1
TYPE_REAL = 2
TYPE_KEYFRAME = 3

# Define classes that are used in this script
class PartitionedSequence(object):

    def __init__(self, step_count, executor):
        # type: (int, PartitioningExecutor) -> None
        self.step_count = step_count
        self.executor = executor
        self._frames = [executor.create_frame(i, step_count) for i in range(step_count + 1)]

    def execute(self):
        self.partition_execute(self.step_count, 0)

    def partition_execute(self, n, start_frame_index, **kwargs):
        if n == 1:
            self.leaf_execute(self._frames[start_frame_index], self._frames[start_frame_index+1], **kwargs)
        else:
            self.branch_execute(n, start_frame_index, **kwargs)

    def leaf_execute(self, start_frame, end_frame, **kwargs):
        self.executor.execute_step(start_frame, end_frame, self)

    def branch_execute(self, n, start_frame_index, **kwargs):
        raise NotImplementedError()

    def partition(self, n, start_frame_index):
        self.executor.partition(n, self._frames[start_frame_index], self._frames[start_frame_index + n],
                                self._frames[start_frame_index + n // 2])

    def __getitem__(self, item):
        return self._frames[item]

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        return self._frames.__iter__()

class StaggeredSequence(PartitionedSequence):

    def __init__(self, step_count, executor):
        PartitionedSequence.__init__(self, step_count, executor)

    def branch_execute(self, n, start_frame_index, **kwargs):
        self.partition(n, start_frame_index)
        self.partition_execute(n//2, start_frame_index)
        self.partition_execute(n//2, start_frame_index+n//2)

class SkipSequence(PartitionedSequence):

    def __init__(self, sequence_length, executor):
        PartitionedSequence.__init__(self, sequence_length, executor)
        for i in range(1, sequence_length):
            if i != sequence_length//2:
                self._frames[i] = None

    def branch_execute(self, n, start_frame_index, **kwargs):
        self.partition(n, start_frame_index)


class LinearSequence(PartitionedSequence):

    def __init__(self, step_count, executor):
        PartitionedSequence.__init__(self, step_count, executor)

    def execute(self):
        for frame1, frame2 in zip(self._frames[:-1], self._frames[1:]):
            self.leaf_execute(frame1, frame2)

    def branch_execute(self, n, start_frame_index, **kwargs):
        raise AssertionError()

class PDE(object):

    def __init__(self):
        self.fields = {}
        self.scalars = {}

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        raise NotImplementedError(self)

    def placeholder_state(self, world, age):
        with struct.VARIABLES:
            with struct.DATA:
                placeholders = placeholder(world.state.staticshape)
        result = struct.map_item(State.age, lambda _: age, placeholders)
        return result

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        raise NotImplementedError(self)

def property_name(trace): 
    return trace.name

def collect_placeholders_channels(placeholder_states, trace_to_channel=property_name):
    if trace_to_channel is None:
        trace_to_channel = property_name
    placeholders = []
    channels = []

    for i, state in enumerate(placeholder_states):
        if state is not None:
            traces = struct.flatten(state, trace=True)
            for trace in traces:
                if isplaceholder(trace.value):
                    placeholders.append(trace.value)
                    channel = trace_to_channel(trace)
                    channels.append(consecutive_frames(channel, len(placeholder_states))[i])
    return placeholders, tuple(channels)

class BurgersPDE(PDE):

    def __init__(self, domain, viscosity, dt):
        PDE.__init__(self)
        self.domain = domain
        self.viscosity = viscosity
        self.dt = dt

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        world.reset(world.batch_size, add_default_objects=False)
        u0 = BurgersVelocity(self.domain, viscosity=self.viscosity, batch_size=world.batch_size, name='burgers')
        world.add(u0, ReplacePhysics())

    def target_matching_loss(self, target_state, actual_state):
        # Only needed for supervised initialization
        diff = target_state.burgers.velocity.data - actual_state.burgers.velocity.data
        loss = math.l2_loss(diff)
        return loss

    def total_force_loss(self, states):
        l2 = []
        l1 = []
        for s1, s2 in zip(states[:-1], states[1:]):
            natural_evolution = Burgers().step(s1.burgers, dt=self.dt)
            diff = s2.burgers.velocity - natural_evolution.velocity
            l2.append(math.l2_loss(diff.data))
            l1.append(math.l1_loss(diff.data))
        l2 = math.sum(l2)
        l1 = math.sum(l1)
        self.scalars["Total Force"] = l1
        return l2

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        b1, b2 = initial_worldstate.burgers, target_worldstate.burgers
        with tf.variable_scope("OP%d" % n):
            predicted_tensor = op_resnet(b1.velocity.data, b2.velocity.data)
        new_field = b1.copied_with(velocity=predicted_tensor, age=(b1.age + b2.age) / 2.)
        return initial_worldstate.state_replaced(new_field)

def op_resnet(initial, target, training=True, trainable=True, reuse=tf.AUTO_REUSE):
    # Set up Tensor y
    y = tf.concat([initial, target], axis=-1)
    downres_padding = sum([2 ** i for i in range(5)])  # 1+2+4+8+16=31
    y = tf.pad(y, [[0, 0], [0, downres_padding], [0, 0]], mode="CONSTANT", constant_values=0)
    resolutions = [y]
    # Add 1D convolution layers with varying kernel sizes:  1x conv1d(y, kernel), 2x residual block (2x con1d 2x ReLu)
    for i, filters in enumerate([4, 8, 16, 16, 16]):
        y = tf.layers.conv1d(
            resolutions[0], filters, kernel_size=2, strides=2, activation=tf.nn.relu, padding="valid", name="downconv_%d" % i, trainable=trainable, reuse=reuse
        )
        for j in range(2):
            y = residual_block_1d(y, filters, name="downrb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)
    # Add 1D convolution layers with equal kernel size:  1x conv1d(y, kernel)
    for j, nb_channels in enumerate([16, 16, 16]):
        y = residual_block_1d(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)
    # Add
    for i, resolution_data in enumerate(resolutions[1:]):
        y = math.upsample2x(y)
        res_in = resolution_data[:, 0:y.shape[1], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions) - 2:
            y = tf.pad(tensor=y, paddings=[[0, 0], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv1d(
                y, filters=16, kernel_size=2, strides=1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse
            )
            for j, nb_channels in enumerate([16, 16]):
                y = residual_block_1d(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(tensor=y, paddings=[[0, 0], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv1d(
                y, filters=1, kernel_size=2, strides=1, activation=None, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse
            )
    return y

class ReplacePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, dependencies=[StateDependency("next_state_prediction", "next_state_prediction", single_state=True, blocking=True)])

    def step(self, state, dt=1.0, next_state_prediction=None):
        return next_state_prediction.prediction.burgers

class PartitioningExecutor(object):

    def create_frame(self, index, step_count):
        return SeqFrame(index, type=TYPE_KEYFRAME if index == 0 or index == step_count else TYPE_UNKNOWN)

    def execute_step(self, initial_frame, target_frame, sequence):
        # type: (SeqFrame, SeqFrame, int) -> None
        print("Execute -> %d" % (initial_frame.index + 1))
        assert initial_frame.type >= TYPE_REAL
        target_frame.type = max(TYPE_REAL, target_frame.type)

    def partition(self, n, initial_frame, target_frame, center_frame):
        # type: (int, SeqFrame, SeqFrame, SeqFrame) -> None
        print("Partition length %d sequence (from %d to %d) at frame %d" % (n, initial_frame.index, target_frame.index, center_frame.index))
        assert initial_frame.type != TYPE_UNKNOWN and target_frame.type != TYPE_UNKNOWN
        center_frame.type = TYPE_PLANNED

class SeqFrame(object):

    def __init__(self, index, type=TYPE_UNKNOWN):
        self.index = index
        self.type = type

    def next(self):
        return self.index + 1

    def __repr__(self):
        return "Frame#%d" % self.index

class StateFrame(SeqFrame):

    def __init__(self, index, type):
        SeqFrame.__init__(self, index, type)
        self.worldstate = None

    def __getitem__(self, item):
        if isinstance(item, StateProxy):
            item = item.state
        return self.worldstate[item]


class PDEExecutor(PartitioningExecutor):

    def __init__(self, world, pde, target_state, trainable_networks, dt):
        self.world = world
        self.pde = pde
        self.worldsteps = 0
        self.next_state_prediction = NextStatePrediction(None)
        self.world.add(self.next_state_prediction)
        self.target_state = target_state
        self.trainable_networks = trainable_networks
        self.dt = dt

    def create_frame(self, index, step_count):
        frame = StateFrame(index, type=TYPE_KEYFRAME if index == 0 or index == step_count else TYPE_UNKNOWN)
        if index == 0:
            frame.worldstate = self.world.state
        if index == step_count:
            frame.worldstate = self.target_state
        return frame

    def execute_step(self, initial_frame, target_frame, sequence):
        PartitioningExecutor.execute_step(self, initial_frame, target_frame, sequence)
        assert initial_frame.index == self.worldsteps == target_frame.index - 1
        ws = initial_frame.worldstate
        if isinstance(sequence, LinearSequence):
            predicted_ws = self.target_state
        else:
            assert target_frame.worldstate is not None
            predicted_ws = target_frame.worldstate
        target_pred = ws[self.next_state_prediction].copied_with(prediction=predicted_ws)
        initial_frame.worldstate = ws.state_replaced(target_pred)
        self.world.state = initial_frame.worldstate
        self.world.step(dt=self.dt)
        self.worldsteps += 1
        if target_frame is sequence[-1]:
            self.world.remove(NextStatePrediction)
        target_frame.worldstate = self.world.state

    def partition(self, n, initial_frame, target_frame, center_frame):
        PartitioningExecutor.partition(self, n, initial_frame, target_frame, center_frame)
        center_frame.worldstate = self.pde.predict(n, initial_frame.worldstate, target_frame.worldstate, trainable='OP%d' % n in self.trainable_networks)

        if center_frame.index == self.worldsteps + 1:
            assert center_frame.worldstate is not None
            old_state = self.next_state_prediction
            self.next_state_prediction = self.next_state_prediction.copied_with(prediction=center_frame.worldstate)
            initial_frame.worldstate = self.world.state.state_replaced(self.next_state_prediction)

    def load(self, max_n, checkpoint_dict, preload_n, session, logf):
        # Control Force Estimator (CFE)
        if 'CFE' in checkpoint_dict:
            ik_checkpoint = os.path.expanduser(checkpoint_dict['CFE'])
            logf("Loading CFE from %s..." % ik_checkpoint)
            session.restore(ik_checkpoint, scope='CFE')
        # Observation Predictors (OP)
        n = 2
        while n <= max_n:
            if n == max_n and not preload_n: return
            checkpoint_path = None
            i = n
            while not checkpoint_path:
                if "OP%d"%i in checkpoint_dict:
                    checkpoint_path = os.path.expanduser(checkpoint_dict["OP%d"%i])
                else:
                    i //= 2
            if i == n:
                logf("Loading OP%d from %s..." % (n, checkpoint_path))
                session.restore(checkpoint_path, scope="OP%d" % n)
            else:
                logf("Loading OP%d from OP%d checkpoint from %s..." % (n, i, checkpoint_path))
                session.restore.restore_new_scope(checkpoint_path, "OP%d" % i, "OP%d" % n)
            n *= 2


    def load_all_from(self, max_n, ik_checkpoint, sm_checkpoint, sm_n, session, logf):
        # IK
        logf("Loading IK checkpoint from %s..." % ik_checkpoint)
        session.restore(ik_checkpoint, scope="ik")
        # SM
        n = 2
        while n <= max_n:
            source_n = sm_n(n) if callable(sm_n) else sm_n
            logf("Loading SM%d weights from SM%d checkpoint from %s..." % (n, source_n, sm_checkpoint))
            session.restore_new_scope(sm_checkpoint, "sm%d" % source_n, "sm%d" % n)
            n *= 2


@struct.definition()
class NextStatePrediction(State):

    def __init__(self, prediction, tags=('next_state_prediction',), name='next', **kwargs):
        State.__init__(self, **struct.kwargs(locals()))

    @struct.variable()
    def prediction(self, prediction):
        assert prediction is None or isinstance(prediction, StateCollection)
        return prediction

    def __repr__(self):
        return self.__class__.__name__

class ControlTraining(LearningApp):

    def __init__(self, n, pde, datapath, val_range, train_range,
                 trace_to_channel=None,
                 obs_loss_frames=(-1,),
                 trainable_networks=('CFE', 'OP2'),
                 sequence_class=StaggeredSequence,
                 batch_size=16,
                 view_size=16,
                 learning_rate=1e-3,
                 learning_rate_half_life=1000,
                 dt=1.0,
                 new_graph=True):
        """
        :param n:
        :param pde:
        :param datapath:
        :param sequence_matching:
        :param train_cfe:
        """
        if new_graph:
            tf.reset_default_graph()
        LearningApp.__init__(self, 'Control Training', 'Train PDE control: OP / CFE', training_batch_size=batch_size,
                             validation_batch_size=batch_size, learning_rate=learning_rate, stride=50)
        self.initial_learning_rate = learning_rate
        self.learning_rate_half_life = learning_rate_half_life
        if n <= 1:
            sequence_matching = False
        diffphys = sequence_class is not None
        if sequence_class is None:
            assert 'CFE' not in trainable_networks, 'CRE training requires a sequence_class.'
            assert len(obs_loss_frames) > 0, 'No loss provided (no obs_loss_frames and no sequence_class).'
            sequence_class = SkipSequence
        self.n = n
        self.dt = dt
        self.data_path = datapath
        self.checkpoint_dict = None
        self.info('Sequence class: %s' % sequence_class)

        # --- Set up PDE sequence ---
        world = World(batch_size=batch_size)
        pde.create_pde(world, 'CFE' in trainable_networks, sequence_class != LinearSequence)  # TODO BATCH_SIZE=None
        world.state = pde.placeholder_state(world, 0)
        self.add_all_fields('GT', world.state, 0)
        target_state = pde.placeholder_state(world, n*dt)
        self.add_all_fields('GT', target_state, n)
        in_states = [world.state] + [None] * (n-1) + [target_state]
        for frame in obs_loss_frames:
            if in_states[frame] is None:
                in_states[frame] = pde.placeholder_state(world, frame*self.dt)
        # --- Execute sequence ---
        executor = self.executor = PDEExecutor(world, pde, target_state, trainable_networks, self.dt)
        sequence = self.sequence = sequence_class(n, executor)
        sequence.execute()
        all_states = self.all_states = [frame.worldstate for frame in sequence if frame is not None]
        # --- Loss ---
        loss = 0
        reg = None
        if diffphys:
            target_loss = pde.target_matching_loss(target_state, sequence[-1].worldstate)
            self.info('Target loss: %s' % target_loss)
            if target_loss is not None:
                loss += target_loss
            reg = pde.total_force_loss([state for state in all_states if state is not None])
            self.info('Force loss: %s' % reg)
        for frame in obs_loss_frames:
            supervised_loss = pde.target_matching_loss(in_states[frame], sequence[frame].worldstate)
            if supervised_loss is not None:
                self.info('Supervised loss at frame %d: %s' % (frame, supervised_loss))
                self.add_scalar('GT_obs_%d' % frame, supervised_loss)
                self.add_all_fields('GT', in_states[frame], frame)
                loss += supervised_loss
        self.info('Setting up loss')
        if loss is not 0:
            self.add_objective(loss, 'Loss', reg=reg)
        for name, scalar in pde.scalars.items():
            self.add_scalar(name, scalar)
        # --- Training data ---
        self.info('Preparing data')
        placeholders, channels = collect_placeholders_channels(in_states, trace_to_channel=trace_to_channel)
        data_load_dict = {p: c for p, c in zip(placeholders, channels)}
        self.set_data(data_load_dict,
                      val=None if val_range is None else Dataset.load(datapath, val_range),
                      train=None if train_range is None else Dataset.load(datapath, train_range))
        # --- Show all states in GUI ---
        for i, (placeholder, channel) in enumerate(zip(placeholders, channels)):
            def fetch(i=i): return self.viewed_batch[i]
            self.add_field('%s %d' % (channel, i), fetch)
        for i, worldstate in enumerate(all_states):
            self.add_all_fields('Sim', worldstate, i)
        for name, field in pde.fields.items():
            self.add_field(name, field)

    def add_all_fields(self, prefix, worldstate, index):
        with struct.unsafe():
            fields = struct.flatten(struct.map(lambda x: x, worldstate, trace=True))
        for field in fields:
            name = '%s[%02d] %s' % (prefix, index, field.path())
            if field.value is not None:
                self.add_field(name, field.value)
            # else:
            #     self.info('Field %s has value None' % name)

    def load_checkpoints(self, checkpoint_dict):
        if not self.prepared:
            self.prepare()
        self.checkpoint_dict = checkpoint_dict
        self.executor.load(self.n, checkpoint_dict, preload_n=True, session=self.session, logf=self.info)

    def action_save_model(self):
        self.save_model()

    def step(self):
        if self.learning_rate_half_life is not None:
            self.float_learning_rate = self.initial_learning_rate * 0.5 ** (self.steps / float(self.learning_rate_half_life))
        LearningApp.step(self)

    def infer_all_frames(self, data_range):
        dataset = Dataset.load(self.data_path, data_range)
        reader = BatchReader(dataset, self._channel_struct)
        batch = reader[0:len(reader)]
        feed_dict = self._feed_dict(batch, True)
        inferred = self.session.run(self.all_states, feed_dict=feed_dict)
        return inferred

    def infer_scalars(self, data_range):
        dataset = Dataset.load(self.data_path, data_range)
        reader = BatchReader(dataset, self._channel_struct)
        batch = reader[0:len(reader)]
        feed_dict = self._feed_dict(batch, True)
        scalar_values = self.session.run(self.scalars, feed_dict, summary_key='val', merged_summary=self.merged_scalars, time=self.steps)
        scalar_values = {name: value for name, value in zip(self.scalar_names, scalar_values)}
        return scalar_values

# Define variables
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

N_ENVS = 10                         # On how many environments to train in parallel, load balancing
FINAL_REWARD_FACTOR = STEP_COUNT    # Penalty for not reaching the goal state
STEPS_PER_ROLLOUT = STEP_COUNT * 10 # How many steps to collect per environment between agent updates
N_EPOCHS = 10                       # How many epochs to perform during each agent update
RL_LEARNING_RATE = 1e-4             # Learning rate for agent updates
RL_BATCH_SIZE = 128                 # Batch size for agent updates
RL_ROLLOUTS = 500                  # Number of iterations for RL training

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

DP_TRAINING_ITERATIONS = 1000  # Change this to change training duration
# Note: Original value was 10000 (ten thousand)

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
