# TODO: Simplify this version

# Use the following versions of PhiFlow and stable-baselines3
# !pip install stable-baselines3==1.1 phiflow==1.5.1

import time, csv, os, shutil
from phi.flow import *
from phi.tf.flow import *



# PREPARATION: DEFINE CLASSES

# Custom sequence class (used in ControlTraining; uses PDEExecutor)
class PartitionedSequence(object):

    def __init__(self, step_count, executor):
        self.step_count = step_count
        self.executor = executor
        self._frames = [executor.create_frame(i, step_count) for i in range(step_count + 1)]

    # Used in ControlTraining
    def execute(self):
        self.partition_execute(self.step_count, 0)

    # Used in this class and StaggeredSequence
    def partition_execute(self, n, start_frame_index, **kwargs):
        if n == 1:
            self.leaf_execute(self._frames[start_frame_index], self._frames[start_frame_index+1], **kwargs)
        else:
            self.branch_execute(n, start_frame_index, **kwargs)

    # Used in this class (uses PDEExecutor)
    def leaf_execute(self, start_frame, end_frame, **kwargs):
        self.executor.execute_step(start_frame, end_frame, self)

    # Used in this class and StaggeredSequence
    def branch_execute(self, n, start_frame_index, **kwargs):
        raise NotImplementedError()

    # Used in StaggeredSequence (uses PDEExecutor)
    def partition(self, n, start_frame_index):
        self.executor.partition(n, self._frames[start_frame_index], self._frames[start_frame_index + n],
                                self._frames[start_frame_index + n // 2])

    def __getitem__(self, item):
        return self._frames[item]

#    def __len__(self):
#        return len(self._frames)

#    def __iter__(self):
#        return self._frames.__iter__()

# More specific custom sequence class (used along PartitionedSequence above)
class StaggeredSequence(PartitionedSequence):

    def __init__(self, step_count, executor):
        PartitionedSequence.__init__(self, step_count, executor)

    # Used in Partiotened Sequence
    def branch_execute(self, n, start_frame_index, **kwargs):
        self.partition(n, start_frame_index)
        self.partition_execute(n//2, start_frame_index)
        self.partition_execute(n//2, start_frame_index+n//2)

# Custom function that returns placeholder and channels in ControlTraining class (see below)
def collect_placeholders_channels(placeholder_states, trace_to_channel=lambda trace: 'burgers_velocity'):
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

# Custom class for actual Burger's PDE (used in ControlTraining and in PDEExecutor)
class BurgersPDE():

    def __init__(self, domain, viscosity, dt):
        self.fields = {}
        self.scalars = {}
        self.domain = domain
        self.viscosity = viscosity
        self.dt = dt

    # This creates the Burger's PDE (used in ControlTraining)
    def create_pde(self, world, control_trainable, constant_prediction_offset):
        world.reset(world.batch_size, add_default_objects=False)
        u0 = BurgersVelocity(self.domain, viscosity=self.viscosity, batch_size=world.batch_size, name='burgers')
        world.add(u0, ReplacePhysics())

    # This is needed to set up the sequence in the PDE (used in ControlTraining)
    def placeholder_state(self, world, age):
        with struct.VARIABLES:
            with struct.DATA:
                placeholders = placeholder(world.state.staticshape)
        result = struct.map_item(State.age, lambda _: age, placeholders)
        return result

    # Target Loss (used in ControlTraining)
    def target_matching_loss(self, target_state, actual_state):
        # Only needed for supervised initialization
        diff = target_state.burgers.velocity.data - actual_state.burgers.velocity.data
        loss = math.l2_loss(diff)
        return loss

    # Force Loss (used in ControlTraining)
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

    # Needed for the partition function in PDEExecutor
    # Used in PDEExecutor (partition function); Uses op_resnet function below
    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        b1, b2 = initial_worldstate.burgers, target_worldstate.burgers
        with tf.variable_scope("OP%d" % n):
            predicted_tensor = op_resnet(b1.velocity.data, b2.velocity.data)
        new_field = b1.copied_with(velocity=predicted_tensor, age=(b1.age + b2.age) / 2.)
        return initial_worldstate.state_replaced(new_field)

# Custom function needed to predict tensor in BurgersPDE class (see above)
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

# Custom class needed to create the PDE in BurgersPDE (see above) (uses NextStatePrediction)
class ReplacePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, dependencies=[StateDependency("next_state_prediction", "next_state_prediction", single_state=True, blocking=True)])

    # Used in BurgersPDE; Uses NextStatePrediction
    def step(self, state, dt=1.0, next_state_prediction=None):
        return next_state_prediction.prediction.burgers

# Custom class needed for PDEExecutor and the Sequence classes
class StateFrame():

    def __init__(self, index):
        self.worldstate = None
        self.index = index

#    def next(self):
#        return self.index + 1

#    def __repr__(self):
#        return "Frame#%d" % self.index

    def __getitem__(self, item):
        if isinstance(item, StateProxy):
            item = item.state
        return self.worldstate[item]

# Custom class to execute PDEs (sed in PartitionedSequence; uses BurgersPDE)
class PDEExecutor():

    def __init__(self, world, pde, target_state, trainable_networks, dt):
        self.world = world
        self.pde = pde
        self.worldsteps = 0
        self.next_state_prediction = NextStatePrediction(None)
        self.world.add(self.next_state_prediction)
        self.target_state = target_state
        self.trainable_networks = trainable_networks
        self.dt = dt

    # Creates frames for the PDE (used in PartitionedSequence)
    def create_frame(self, index, step_count):
        frame = StateFrame(index)
        if index == 0:
            frame.worldstate = self.world.state
        if index == step_count:
            frame.worldstate = self.target_state
        return frame

    # Executes (a step in) the PDE (used in PartitionedSequence)
    def execute_step(self, initial_frame, target_frame, sequence):
        assert initial_frame.index == self.worldsteps == target_frame.index - 1
        ws = initial_frame.worldstate
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

    # Needed for the PartitionedSequence class (used in PartitionedSequence; uses predict in BurgersPDE)
    def partition(self, n, initial_frame, target_frame, center_frame):
        center_frame.worldstate = self.pde.predict(n, initial_frame.worldstate, target_frame.worldstate, trainable='OP%d' % n in self.trainable_networks)

        if center_frame.index == self.worldsteps + 1:
            assert center_frame.worldstate is not None
            old_state = self.next_state_prediction
            self.next_state_prediction = self.next_state_prediction.copied_with(prediction=center_frame.worldstate)
            initial_frame.worldstate = self.world.state.state_replaced(self.next_state_prediction)

# Custom class used to execute step and for partions in PDEExecutor and ReplacePhysics
@struct.definition()
class NextStatePrediction(State):

    def __init__(self, prediction, tags=('next_state_prediction',), name='next', **kwargs):
        State.__init__(self, **struct.kwargs(locals()))

    # Actual prediction (Used in ReplacePhysics -> BurgersPDE)
    @struct.variable()
    def prediction(self, prediction):
        assert prediction is None or isinstance(prediction, StateCollection)
        return prediction

#    def __repr__(self):
#        return self.__class__.__name__

# Main custom class that is actually needed in the simulation below
# Note: All other classes seem to be helpers to execute this class and the code in the simulation
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
        # --- Initialize values ---
        if new_graph:
            tf.reset_default_graph()
        LearningApp.__init__(self, 'Control Training', 'Train PDE control: OP / CFE', training_batch_size=batch_size,
                             validation_batch_size=batch_size, learning_rate=learning_rate, stride=50)
        self.initial_learning_rate = learning_rate
        self.learning_rate_half_life = learning_rate_half_life
        self.n = n
        self.dt = dt
        self.data_path = datapath
        self.info('Sequence class: %s' % sequence_class)

        # --- Set up PDE sequence ---
        # Creates PDE & grid
        world = World(batch_size=batch_size)
        pde.create_pde(world, 'CFE' in trainable_networks, sequence_class == StaggeredSequence)  # TODO BATCH_SIZE=None
        world.state = pde.placeholder_state(world, 0)
        self.add_all_fields('GT', world.state, 0)
        target_state = pde.placeholder_state(world, n*dt)
        self.add_all_fields('GT', target_state, n)
        in_states = [world.state] + [None] * (n-1) + [target_state]
#        for frame in obs_loss_frames:
#            if in_states[frame] is None:
#                in_states[frame] = pde.placeholder_state(world, frame*self.dt)

        # --- Execute sequence ---
        # Defines executor and sequence
        executor = self.executor = PDEExecutor(world, pde, target_state, trainable_networks, self.dt)
        sequence = self.sequence = sequence_class(n, executor)
        # Executes sequence
        sequence.execute()
        # Recovers all states
        all_states = self.all_states = [frame.worldstate for frame in sequence if frame is not None]

        # --- Loss ---
        loss = 0
        reg = None
        # Target Loss
        target_loss = pde.target_matching_loss(target_state, sequence[-1].worldstate)
        self.info('Target loss: %s' % target_loss)
        if target_loss is not None:
            loss += target_loss
        # Total Force Loss
        reg = pde.total_force_loss([state for state in all_states if state is not None])
        self.info('Force loss: %s' % reg)
#        for frame in obs_loss_frames:
#            supervised_loss = pde.target_matching_loss(in_states[frame], sequence[frame].worldstate)
#            if supervised_loss is not None:
#                self.info('Supervised loss at frame %d: %s' % (frame, supervised_loss))
#                self.add_scalar('GT_obs_%d' % frame, supervised_loss)
#                self.add_all_fields('GT', in_states[frame], frame)
#                loss += supervised_loss
        # Stores loss
        self.info('Setting up loss')
        if loss is not 0:
            self.add_objective(loss, 'Loss', reg=reg)
        for name, scalar in pde.scalars.items():
            self.add_scalar(name, scalar)

        # --- Training data ---
        # Preparing training data
        self.info('Preparing data')
        placeholders, channels = collect_placeholders_channels(in_states, trace_to_channel=trace_to_channel)
        data_load_dict = {p: c for p, c in zip(placeholders, channels)}
        self.set_data(data_load_dict,
                      val=None if val_range is None else Dataset.load(datapath, val_range),
                      train=None if train_range is None else Dataset.load(datapath, train_range))

    # Stores all given fields
    def add_all_fields(self, prefix, worldstate, index):
        with struct.unsafe():
            fields = struct.flatten(struct.map(lambda x: x, worldstate, trace=True))
        for field in fields:
            name = '%s[%02d] %s' % (prefix, index, field.path())
            if field.value is not None:
                self.add_field(name, field.value)

    # Performs a step in ControlTraining
    def step(self):
        if self.learning_rate_half_life is not None:
            self.float_learning_rate = self.initial_learning_rate * 0.5 ** (self.steps / float(self.learning_rate_half_life))
        LearningApp.step(self)

    # Infers the scalar values in data_range
    def infer_scalars(self, data_range):
        dataset = Dataset.load(self.data_path, data_range)
        reader = BatchReader(dataset, self._channel_struct)
        batch = reader[0:len(reader)]
        feed_dict = self._feed_dict(batch, True)
        scalar_values = self.session.run(self.scalars, feed_dict, summary_key='val', merged_summary=self.merged_scalars, time=self.steps)
        scalar_values = {name: value for name, value in zip(self.scalar_names, scalar_values)}
        return scalar_values



# ACTUAL SIMULATION

# Define variables
DOMAIN = Domain([32], box=box[0:1])     # Size and shape of the fields
VISCOSITY = 0.003
STEP_COUNT = 32                         # Trajectory length
DT = 0.03

DATA_PATH = 'forced-burgers-clash'
TRAIN_RANGE = range(200, 1000)
VAL_RANGE = range(100, 200)
N_EPOCHS = 10

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

