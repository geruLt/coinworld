import numpy as np
from tqdm import tqdm
import time
from collections import Counter

from models.dqn import DQNAgent
from coinworld import coinworld

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


## Constants
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x128LSTM-2x64DENSE'
MIN_REWARD = 90  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 30  # episodes
SHOW_PREVIEW = False

# Env and agent
env = coinworld()
agent = DQNAgent()

## Training Loop
ep_rewards = []
ep_changes = []

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    episode_change = 0
    step = 1
    actions = []

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, change, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        episode_change += change
        actions.append(action)
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state,action,reward,new_state,done))
        agent.train(done, step)

        current_state = new_state
        step += 1
    if episode_change < 1000 or episode_change > 1000:
        text_file = open("Debug.txt", "w")
        text_file.write(env.debug)
        text_file.close()
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    ep_changes.append(episode_change)
    print('reward %.2f' % episode_reward, 'gain %.2f' % episode_change,
          'actions: ', Counter(actions), '10 eps gain %.3f' %
                                                     np.sum(ep_changes[-10:]))

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])


        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'saved_models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
