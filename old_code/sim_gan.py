import logging
import numpy as np

from blocks.algorithms import Adam, RMSProp
from blocks.bricks import MLP, Tanh, Identity, BatchNormalizedMLP, LeakyRectifier
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from mujoco_py.mjtypes import POINTER, c_double
# from rllab.algos.trpo import TRPO
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.misc.instrument import run_experiment_lite
import theano.tensor as T


from utils import RLMainLoop, Buffer
from models.gan import GAN, WeightClipping

# logging.basicConfig(filename='example.log',
#     level=logging.DEBUG,
#     format='%(message)s')
#
# logging.getLogger().addHandler(logging.StreamHandler())
# logger = logging.getLogger(__name__)

## Definiing the environnement
coeff = 0.85

env = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)
env2 = normalize(GymEnv('Swimmer-v1', force_reset=True), normalize_obs=True)

# The second environnement models the real world
#env2.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents
env2.wrapped_env.env.env.model.opt.gravity = np.array([0, 0, -9.81*coeff]).ctypes.data_as(POINTER(c_double*3)).contents


## Defining the buffer
observation_dim = int(env.observation_space.shape[0])
action_dim = int(env.action_space.shape[0])
rng = np.random.RandomState(seed=23)
max_steps = 10000
history = 2

buffer_ = Buffer(observation_dim, action_dim, rng, history, max_steps)

## Defining the generative model
input_dim = (history+1)*observation_dim + history*action_dim
h = 64
LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-4
BETA1 = 0.5

# Doesn't make sense to use batch_norm in online training
# so far the environnement and generative model run synchronously

actions_var = env.action_space.new_tensor_variable(
    'actions',
    extra_dims=1
)
obs_prev = env.observation_space.new_tensor_variable(
    'previous_obs',
    extra_dims=1
)
obs_sim = env.observation_space.new_tensor_variable(
    'observation_sim',
    extra_dims=0
)
obs_real = env.observation_space.new_tensor_variable(
    'observation_real',
    extra_dims=0
)
context = T.concatenate([actions_var.flatten(), obs_prev.flatten()])

G = MLP(
    activations=[LeakyRectifier(), LeakyRectifier(), Tanh()],
    dims=[input_dim, h, h, observation_dim],
    name='generator'
)
D = MLP(
    activations=[LeakyRectifier(), LeakyRectifier(), Identity()],
    dims=[input_dim, h, h, 1],
    name='discriminator'
)

generative_model = GAN(G, D, alpha=0, weights_init=IsotropicGaussian(std=0.02, mean=0),
    biases_init=Constant(0.01), name='GAN')
generative_model.initialize()
cg = ComputationGraph(generative_model.losses(context, obs_sim, obs_real))

model = Model(cg.outputs)
discriminator_loss, generator_loss = model.outputs

auxiliary_variables = [u for u in model.auxiliary_variables if 'norm' not in u.name]
auxiliary_variables.extend(VariableFilter(theano_name='squared_error')(model.variables))

step_rule = Adam(learning_rate=LEARNING_RATE, beta1=BETA1)
algorithm = generative_model.algorithm(discriminator_loss, generator_loss, step_rule, step_rule)

d = {
    "env": env,
    "env2": env2,
    "buffer_": buffer_,
    "history": history,
    "render": False,
    "episode_len": 10,
    "trajectory_len": 100,
}

extensions = [
    #Timing(),
    FinishAfter(after_n_epochs=10000),
    WeightClipping(after_batch=True),
    TrainingDataMonitoring(
        model.outputs+auxiliary_variables,
        # prefix="train",
        after_epoch=True),
    # Checkpoint('generative_model.tar', after_n_epochs=10, after_training=True,
    #            use_cpickle=True),
    Printing()
]

main_loop = RLMainLoop(
    algorithm,
    d,
    model=model,
    extensions=extensions,
)

main_loop.run()