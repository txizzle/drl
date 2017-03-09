import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# Plotting for Q1, performance on Pong for lr=1 on images for 4.2m steps
pixel_data = pickle.load(open('atari_lr1/pong_step5000000_data.pkl','rb'))
pixel_t = pixel_data['t_log']
pixel_mean_rewards = pixel_data['mean_reward_log']
pixel_best_rewards = pixel_data['best_mean_log']

pixel_plot= plt.figure()
pixel_mean_rew, = plt.plot(pixel_t, pixel_mean_rewards, label='Mean 100-Episode Reward')
pixel_best_rew, = plt.plot(pixel_t, pixel_best_rewards, label='Best Mean Reward')
plt.suptitle('Q-Learning Performance on Pong with Pixels', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=4)
pp = PdfPages('lr1_images_plot.pdf')
pp.savefig(pixel_plot)
pp.close()

# Plotting for Q2, performance on Pong on RAM for 1M Steps with Various Learning Rates
lr_01_data = pickle.load(open('ram_lr01/ram_lr0.1_1000000_data.pkl','rb'))
lr_01_t = lr_01_data['t_log']
lr_01_mean_rewards = lr_01_data['mean_reward_log']
lr_01_best_rewards = lr_01_data['best_mean_log']

lr_1_data = pickle.load(open('ram_lr1/ram_lr1_1000000_data.pkl','rb'))
lr_1_t = lr_1_data['t_log']
lr_1_mean_rewards = lr_1_data['mean_reward_log']
lr_1_best_rewards = lr_1_data['best_mean_log']


lr_10_data = pickle.load(open('ram_lr10/ram_lr10_1000000_data.pkl','rb'))
lr_10_t = lr_10_data['t_log']
lr_10_mean_rewards = lr_10_data['mean_reward_log']
lr_10_best_rewards = lr_10_data['best_mean_log']

lr_100_data = pickle.load(open('ram_lr100/ram_lr100.0_1000000_data.pkl','rb'))
lr_100_t = lr_100_data['t_log']
lr_100_mean_rewards = lr_100_data['mean_reward_log']
lr_100_best_rewards = lr_100_data['best_mean_log']

all_plot= plt.figure()
lr100_mean_rew, = plt.plot(lr_100_t, lr_100_mean_rewards, label='LR = 100')
lr10_mean_rew, = plt.plot(lr_10_t, lr_10_mean_rewards, label='LR = 10')
lr1_mean_rew, = plt.plot(lr_1_t, lr_1_mean_rewards, label='LR = 1')
lr01_mean_rew, = plt.plot(lr_01_t, lr_01_mean_rewards, label='LR = 0.1')
plt.suptitle('Learning Rate vs. Q-Learning Performance for Pong w/ RAM', fontsize=16)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=2)
pp = PdfPages('lr_plot.pdf')
pp.savefig(all_plot)
pp.close()