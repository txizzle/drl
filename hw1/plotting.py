import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

demonstrations = [10, 15, 20, 25, 30]
rewards = [4323.5394949030442, 3886.4435774629064, 4164.2909894785198, 4477.5970902037734, 4504.2251490910221]
stds = [71.027370802334829, 68.37769084214959, 97.309782342169484, 73.664794770305548, 60.799254994971939]

demonstrations_plot = plt.figure()
plt.plot(demonstrations, rewards)
plt.errorbar(demonstrations, rewards, yerr=stds, fmt='o')
plt.suptitle('Behavorial Cloning: Demonstrations vs. Reward', fontsize=20)
plt.xlabel('Number of Expert Demonstrations')
plt.ylabel('Mean Reward')
plt.xlim([5,35])
print('lol')
pp = PdfPages('demonstrations_plot.pdf')
pp.savefig(demonstrations_plot)
pp.close()

iteration = [0, 1, 2, 3, 4]
dagger_rewards = [4296.023579651287, 4525.5670792723868, 4451.7678894108849, 4601.1988741683672, 4657.2109483279946]
dagger_stds = [412.88923326493426, 56.972266843540197, 291.26113241063399, 80.998573493581489, 86.64672308015588]
dagger_plot= plt.figure()
dag, = plt.plot(iteration, dagger_rewards, label='DAgger Policy')
plt.errorbar(iteration, dagger_rewards, yerr=dagger_stds, fmt='o')
plt.suptitle('DAgger Iterations vs. Rewards', fontsize=20)
plt.xlabel('DAgger Iteration')
plt.ylabel('Mean Reward')
plt.xlim([-0.5, 4.5])
plt.ylim([3800, 5000])
expert = plt.axhline(y=4796.2601920496254, color='k', label='Expert Policy')
bc = plt.axhline(y=4477.5970902037734, color='r', label='Behaviorial Cloning')
plt.legend(loc=4)
pp = PdfPages('dagger_plot.pdf')
pp.savefig(dagger_plot)
pp.close()
