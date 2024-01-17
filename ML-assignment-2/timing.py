import matplotlib.pyplot as plt
import os
parallel_time = [36.22031211853027, 48.63858222961426, 88.09971809387207, 128.2973289489746, 555.9124946594238, 1080.489158630371]
non_parallel_time = [7.7457427978515625, 16.391277313232422, 57.35158920288086, 103.50966453552246, 567.4417018890381, 1137.8514766693115]
N = [30, 100, 500, 1000, 5000, 10000]
plt.plot(N,parallel_time)
plt.plot(N,non_parallel_time)
plt.title('parallelization vs non-parallization')
plt.xlabel("Datapoints")
plt.ylabel('Time in ms')
plt.savefig(os.path.join("figures", "Time_analysis_bagging.png"))
plt.show()