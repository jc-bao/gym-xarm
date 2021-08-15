import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

env_id = "XarmPDHandoverNoGoal-v1"
log_dir = 'saved_data/'+env_id
num_cpu = 4
timesteps = 25000

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Xarm_Handover_Dense")
plt.show()