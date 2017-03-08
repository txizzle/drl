import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# Plotting for Q1, performance on Pong for lr=1 on images for 4.2m steps
lr_1_data = pickle.load(open('atari_lr1/pong_step5000000_data.pkl','rb'))
lr_1_t = lr_1_data['t_log']
lr_1_mean_rewards = lr_1_data['mean_reward_log']
lr_1_best_rewards = lr_1_data['best_mean_log']

print(lr_1_data.keys)

lr_1_plot= plt.figure()
lr_1_mean_rew, = plt.plot(lr_1_t, lr_1_mean_rewards, label='Mean 100-Episode Reward')
lr_1_best_rew, = plt.plot(lr_1_t, lr_1_best_rewards, label='Best Mean Reward')
plt.suptitle('Q-Learning Performance on Pong with Pixels', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=4)
pp = PdfPages('lr1_images_plot.pdf')
pp.savefig(lr_1_plot)
pp.close()

# Plotting for Q2, performance on Pong for lr=10 on RAM for XXX steps
lr_10_data = pickle.load(open('ram_lr10/pong_step4200000_data.pkl','rb'))
lr_10_t = lr_10_data['t_log']
lr_10_mean_rewards = lr_10_data['mean_reward_log']
lr_10_best_rewards = lr_10_data['best_mean_log']

lr_10_plot= plt.figure()
mean_rew, = plt.plot(lr_10_t, lr_10_mean_rewards, label='Mean 100-Episode Reward')
best_rew, = plt.plot(lr_10_t, lr_10_best_rewards, label='Best Mean Reward')
plt.suptitle('Q-Learning Performance on Pong with RAM, Learning Rate = 10', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=4)
pp = PdfPages('lr10_ram_plot.pdf')
pp.savefig(lr_10_plot)
pp.close()
# iteration = [0, 1, 2, 3, 4]
# dagger_rewards = [4296.023579651287, 4525.5670792723868, 4451.7678894108849, 4601.1988741683672, 4657.2109483279946]
# dagger_stds = [412.88923326493426, 56.972266843540197, 291.26113241063399, 80.998573493581489, 86.64672308015588]
# dagger_plot= plt.figure()
# dag, = plt.plot(iteration, dagger_rewards, label='DAgger Policy')
# plt.errorbar(iteration, dagger_rewards, yerr=dagger_stds, fmt='o')
# plt.suptitle('DAgger Iterations vs. Rewards', fontsize=20)
# plt.xlabel('DAgger Iteration')
# plt.ylabel('Mean Reward')
# plt.xlim([-0.5, 4.5])
# plt.ylim([3800, 5000])
# expert = plt.axhline(y=4796.2601920496254, color='k', label='Expert Policy')
# bc = plt.axhline(y=4477.5970902037734, color='r', label='Behaviorial Cloning')
# plt.legend(loc=4)
# pp = PdfPages('dagger_plot.pdf')
# pp.savefig(dagger_plot)
# pp.close()

# human_iteration = range(30)
# human_dagger_rewards = [406.5165808392747, 283.15833841520367, 391.0148835655832, 430.95976406320312, 403.83685970861427, 322.57208552037059, 1814.9481575346861, 1575.5064503001963, 6173.5815951610666, 4876.8429164452255, 7593.7542712271461, 9160.3004421308997, 5688.5455353746465, 6925.2393718965368, 5335.8049565231604, 8637.8158092510021, 8672.2098064719139, 10234.348964963558, 8084.9419246883645, 8701.7001021130091, 10417.797955526392, 10370.748452424181, 8411.1555847408017, 10318.279301285051, 10305.766630265178, 10276.94042821201, 8402.664218160684, 8772.3957674275189, 10404.000237326703, 10380.281147198973]
# human_dagger_stds = [59.560134871450067, 19.524873155795504, 54.01747197285043, 116.30982566116887, 96.881933782643543, 23.015717978135374, 1250.3806051375425, 1135.4553290094143, 4280.1960254196456, 4418.6526126895114, 3686.5477403212935, 2262.5885288035943, 4223.472737527396, 4320.839731137612, 1637.4527003824455, 3310.5701755916834, 3168.3082635685464, 325.46066658008624, 2904.4412119666945, 2995.1204139192432, 34.924201056653743, 74.815890874572304, 3926.9450189114232, 117.94482377736936, 101.76302642500366, 124.51855870095501, 3951.1307054791309, 3318.2468786742115, 89.34376543492931, 110.87516981476369]
# human_dagger_plot= plt.figure()
# dag, = plt.plot(human_iteration, human_dagger_rewards, label='DAgger Policy')
# plt.errorbar(human_iteration, human_dagger_rewards, yerr=human_dagger_stds, fmt='o')
# plt.suptitle('Humanoid DAgger Iterations vs. Rewards', fontsize=20)
# plt.xlabel('DAgger Iteration')
# plt.ylabel('Mean Reward')
# plt.xlim([-0.5, 30.5])
# # plt.ylim([3800, 5000])
# expert = plt.axhline(y=10421.023131225815, color='k', label='Expert Policy')
# bc = plt.axhline(y=200, color='r', label='Behaviorial Cloning')
# plt.legend(loc=4)
# pp = PdfPages('human_dagger_plot.pdf')
# pp.savefig(human_dagger_plot)
# pp.close()
