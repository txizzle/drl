# Deep Reinforcement Learning Algorithms
This repository contains implementations of various deep reinforcement learning algorithms completed as part of the Spring 2017 offering of CS 294-112, [UC Berkeley's Deep Reinforcement Learning course](http://rll.berkeley.edu/deeprlcoursesp17/).

*Disclaimer: The code contained in this repository may or may not relate to coursework in future offerings of CS 294-112. The implementations here are provided for educational purposes only; if you are a student in the course, I highly suggest attempting the problems yourself.*

## Dependencies
The dependencies of the algorithms include:
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [MuJoCo](http://www.mujoco.org/) [Paid library, but there is a free student license]

## HW1: Imitation Learning and DAgger on MuJoCo
I implemented [behavior cloning](http://rll.berkeley.edu/deeprlcourse/docs/week_2_lecture_1_behavior_cloning.pdf) on multiple MuJoCo environments. Expert policies produce rollouts that are used as training data for a feedforward neural network. In addition to normal behavior cloning, I also implemented the [DAgger](http://rll.berkeley.edu/deeprlcourse-fa15/docs/2015.10.5.dagger.pdf) algorithm, which performs significantly better. Finally, I varied the number of rollouts used to train the agent, and observed that more rollouts as training data produces better results, as expected.

## HW2: Policy Iteration and Value Iteration for Markov Decision Processes (MDPs)
This is a fairly straightforward implementation of [Policy Iteration and Value Iteration](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf) on a simple gridworld environment. 

## HW3: Deep Q-Networks on Atari Games
I implemented the [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) algorithm on the [Pong](https://gym.openai.com/envs/Pong-v0) Atari environment in the OpenAI Gym. Using pixel data gives better results than using only RAM data.

## HW4: Policy Gradients
I extended the existing discrete [Policy Gradients algorithm](http://karpathy.github.io/2016/05/31/rl/) to [Pendulum](https://gym.openai.com/envs/Pendulum-v0) on OpenAI Gym, a continuous environment. In addition, I used a neural network to learn the value function.

## Final Project
The code for this project has not been released yet, but my writeup can be found [here](http://tedxiao.me/pdf/gans_drl.pdf).
