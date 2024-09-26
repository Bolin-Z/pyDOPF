import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODES = ['global_best', 'top_level_random', 'correspond']
# mode = MODES[2]
# data_num = 3
# dfs = [pd.read_csv(f"./result/{mode}_{i}.csv") for i in range(data_num)]
# fits = [np.array(dfs[i]["history_best"].values) for i in range(data_num)]

# mean_fit = np.zeros(np.shape(fits[0]))
# for i in range(data_num):
#     mean_fit += fits[i]
# mean_fit = mean_fit / data_num

# x = np.arange(len(fits[0]))

# fig, ax = plt.subplots()
# ax.grid()
# ax.plot(x, mean_fit, color="g", label="mean_fit")
# ax.plot(x, fits[1], color="b", label="best")
# ax.legend()
# plt.title(mode)
# plt.show()

dfs = [
    pd.read_csv("./result/global_best_1.csv"),
    pd.read_csv("./result/top_level_random_2.csv"),
    pd.read_csv("./result/correspond_1.csv")
]

fits = [
    np.array(dfs[0]["total_fitness"].values),
    np.array(dfs[1]["history_best"].values),
    np.array(dfs[2]["history_best"].values)
]

x = np.arange(len(fits[0]))

fig, ax = plt.subplots()
ax.grid()
ax.plot(x, fits[0], color="r", label="global_best")
ax.plot(x, fits[1], color="g", label="top_level_random")
ax.plot(x, fits[2], color="b", label="correspond")
ax.legend()
plt.title("compare of best result")
plt.show()