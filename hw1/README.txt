HW1 Imitation Learning
Ted Xiao

To run behavorial cloning or DAgger, this code needs expert demonstrations saved in the format:
Ant-v1_[Number of Rollouts]_data.pkl

To generate those data files, run:
python run_expert.py experts/Ant-v1.pkl Ant-v1 --render --num_rollouts [Number of Rollouts]

To run Behavorial Cloning:
python run_cloning.py experts/Ant-v1.pkl Ant-v1 data/Ant-v1_[Number of Rollouts]_data.pkl --render --num_rollouts [Number of Rollouts]

ex. python run_cloning.py experts/Ant-v1.pkl Ant-v1 data/Ant-v1_[Number of Rollouts]_data.pkl --render --num_rollouts [Number of Rollouts]

To run DAgger:
python dagger.py experts/Ant-v1.pkl Ant-v1 data/Ant-v1_[Number of Rollouts]_data.pkl --render --num_rollouts [Number of Rollouts]

ex. python dagger.py experts/Ant-v1.pkl Ant-v1 data/Ant-v1_[Number of Rollouts]_data.pkl --render --num_rollouts [Number of Rollouts]

To generate the plots:
python plotting.py
