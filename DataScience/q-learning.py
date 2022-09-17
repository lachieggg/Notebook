import gym
import random
import numpy as np

random.seed(1234)

streets = gym.make("Taxi-v3").env
streets.render()

#
# Blue pick up point
# Magenta delivery point
# Rectangle itself represents the taxi
#
# Encode the initial state
# Taxi located at (2,3)
# pickup is at location 2 (B)
# destination is at location 0 (Y)
# YRBG?
#
initial_state = streets.encode(2, 3, 2, 0) 
streets.s = initial_state

# Render the initial state
streets.render()

# Print the initial reward table (Q table)
streets.P[initial_state]

# Create an initial reward table (Q table)
q_table = np.zeros([streets.observation_space.n, streets.action_space.n])

# Set the hyperparameters
learning_rate = 0.4
discount_factor = 0.6
exploration = 0.1
epochs = 4000

# Run a set of learning iterations
for taxi_run in range(epochs):
  state = streets.reset()
  done = False
  
  # Run an individual learning iteration
  # until it completes
  while not done:
    # Generate a random number that could be less than epsilon
    random_value = random.uniform(0, 1) 
    if (random_value < exploration):
      # Explore a random action
      action = streets.action_space.sample() 
    else:
      # Use the action with the highest q-value
      action = np.argmax(q_table[state]) 
            
    next_state, reward, done, info = streets.step(action)
        
    prev_q = q_table[state, action]
    next_max_q = np.max(q_table[next_state])

    # Calculate the Q value for this action
    #
    # Take the next reward or punishment and scale it by the 
    # learning factor
    #
    # Add that to the previous value and scale it down proportional
    # to the learning factor
    #
    delta = learning_rate * (reward + discount_factor * next_max_q)
    curr = (1 - learning_rate) * prev_q
    new_q = curr + delta
    # Set that value in the Q table
    q_table[state, action] = new_q
    
    # Iterate
    state = next_state