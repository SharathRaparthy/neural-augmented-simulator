import pickle
import os
import matplotlib.pyplot as plt
from nas.data import RECORDINGS_PATH
import numpy as np
data = pickle.load(open(os.path.join(RECORDINGS_PATH, "10-lstm-data.pkl"), "rb"))

print(data.keys()
     )  # ['real-posvel', 'actions', 'next-real-posvel', 'next-sim-posvel']
real_posvel = data['real-posvel']
actions = data["actions"]
next_real_posvel = data["next-real-posvel"]
next_sim_posvel = data["next-sim-posvel"]

print(real_posvel.shape)
print(actions.shape)
print(next_real_posvel.shape)
print(next_sim_posvel.shape)

print("real_pos", real_posvel[:, 6].min(), real_posvel[:, 6].max(),
      real_posvel[:, 6].mean())
print("real_vel", real_posvel[:, 6:].min(), real_posvel[:, 6:].max(),
      real_posvel[:, 6:].mean())
print("actions", actions.min(), actions.max(), actions.mean())
print("next_real_pos", next_real_posvel[:, :6].min(),
      next_real_posvel[:, :6].max(), next_real_posvel[:, :6].mean())
print("next_real_vel", next_real_posvel[:, 6:].min(),
      next_real_posvel[:, 6:].max(), next_real_posvel[:, 6:].mean())
print("next_sim_pos", next_sim_posvel[:, :6].min(),
      next_sim_posvel[:, :6].max(), next_sim_posvel[:, :6].mean())
print("next_sim_vel", next_sim_posvel[:, 6:].min(),
      next_sim_posvel[:, 6:].max(), next_sim_posvel[:, 6:].mean())

start = 0
samples = 998
end = start + samples

x = np.arange(start, end)

# print ("first motor REAL:",real_posvel[:,0].min(), real_posvel[:,0].max(), real_posvel[:,0].mean())
# print ("first motor SIM:",next_sim_posvel[:,0].min(), next_sim_posvel[:,0].max(), next_sim_posvel[:,0].mean())
#
# for i in range(6):
#     print (f"actions, motor{i}:", actions[:,i].min(), actions[:,i].max(), actions[:,i].mean())

# ==================== REAL
#
# #
# for i in range(3):
#     plt.plot(x, real_posvel[start:end, i], label=f"motor {i} pos")
#     plt.plot(x, real_posvel[start:end, i+6], label=f"motor {i} vel", linestyle="dashed")
#     plt.plot(x, actions[start:end, i], label=f"motor {i} action", linestyle="dotted")
#
# plt.title(f"100Hz REAL recordings from timestep {start} to {end}")
# plt.legend()
# plt.tight_layout()
#
# plt.show()
#
#
# # ==================== SIM
#
#
# for i in range(3):
#     plt.plot(x, next_sim_posvel[start:end, i], label=f"motor {i} pos")
#     plt.plot(x, next_sim_posvel[start:end, i+6], label=f"motor {i} vel", linestyle="dashed")
#     plt.plot(x, actions[start:end, i], label=f"motor {i} action", linestyle="dotted")
#
# plt.title(f"100Hz SIM recordings from timestep {start} to {end}")
# plt.legend()
# plt.tight_layout()
#
# plt.show()

# ==================== REAL+SIM POS
#
# i = 1 # for example, because motor i=0 is not moving
# plt.plot(x, real_posvel[start:end, i], label=f"REAL: motor {i} pos ")
# plt.plot(x, next_sim_posvel[start:end, i], label=f"NEXT_SIM: motor {i} pos", linestyle="dashed")
# plt.plot(x, next_real_posvel[start:end, i], label=f"NEXT_REAL: motor {i} pos", linestyle="dashed")
# plt.plot(x, actions[start:end, i], label=f"motor {i} action", linestyle="dotted")
#
# plt.title(f"100Hz real/sim recordings from timestep {start} to {end}")
# plt.legend()
# plt.tight_layout()
#
# plt.show()
#
# # ==================== REAL+SIM VEL - IMPORTANT, SIM VELOCITY IS seemingly INVERTED
#
# i = 1 # for example, because motor i=0 is not moving
# plt.plot(x, real_posvel[start:end, 6+i], label=f"REAL: motor {i} vel ")
# plt.plot(x, -next_sim_posvel[start:end, 6+i], label=f"NEXT_SIM: motor {i} vel", linestyle="dashed")
# plt.plot(x, next_real_posvel[start:end, 6+i], label=f"NEXT_REAL: motor {i} vel", linestyle="dashed")
# plt.plot(x, actions[start:end, i], label=f"motor {i} action", linestyle="dotted")
#
# plt.title(f"100Hz real/sim recordings from timestep {start} to {end}")
# plt.legend()
# plt.tight_layout()
#
# plt.show()

# ==================== HISTOGRAMS
#
#
# n_bins = 10
#
# # Generate a normal distribution, center at x=0 and y=5
# pos_real = next_real_posvel[:,:6]
# vel_real = next_real_posvel[:,6:]
# pos_sim = next_sim_posvel[:,:6]
# vel_sim = next_sim_posvel[:,6:]
# fig, axs = plt.subplots(2, 2, tight_layout=True)
#
# # We can set the number of bins with the `bins` kwarg
# axs[0][0].hist(pos_real, bins=n_bins)
# axs[0][0].title.set_text("Positions, Real")
#
# axs[0][1].hist(vel_real, bins=n_bins)
# axs[0][1].title.set_text("Velocities, Real")
#
# axs[1][0].hist(pos_sim, bins=n_bins)
# axs[1][0].title.set_text("Positions, Sim")
#
# axs[1][1].hist(vel_sim, bins=n_bins)
# axs[1][1].title.set_text("Velocities, Sim")
#
# axs[0][0].set_ylim(0,300000)
# axs[0][1].set_ylim(0,300000)
#
# axs[1][0].set_ylim(0,100000)
# axs[1][1].set_ylim(0,100000)
#
# plt.show()

# ==================== HDF5 OUTPUT

import h5py
f = h5py.File(os.path.join(RECORDINGS_PATH, "real-recordings-01-02-10-v1-comp.hdf5"), 'w')

for i in range(3):
    if i == 0:
        dataset = "01"
    if i == 1:
        dataset = "10"
    if i == 2:
        dataset = "02"
    data = pickle.load(
        open(os.path.join(RECORDINGS_PATH, f"{dataset}-lstm-data.pkl"), "rb"))

    real_posvel = data['real-posvel']
    actions = data["actions"]
    next_real_posvel = data["next-real-posvel"]
    next_sim_posvel = data["next-sim-posvel"]

    real_posvel = np.clip(real_posvel, -1, 1).reshape((-1, 1000, 12))
    actions = actions.reshape((-1, 1000, 6))
    next_real_posvel = np.clip(next_real_posvel, -1, 1).reshape((-1, 1000, 12))

    # invert sim velocities. Because the plots reflect that they are the inverse
    next_sim_posvel[:, 6:] = next_sim_posvel[:, 6:] * -1
    next_sim_posvel = np.clip(next_sim_posvel, -1, 1).reshape((-1, 1000, 12))

    # cut out last element, because not connected
    real_posvel = real_posvel[:,1:999,:]
    actions = actions[:,1:999,:]
    next_real_posvel = next_real_posvel[:,1:999,:]
    next_sim_posvel = next_sim_posvel[:,1:999,:]

    print(real_posvel.shape)
    print(actions.shape)
    print(next_real_posvel.shape)
    print(next_sim_posvel.shape)


    gr = f.create_group(dataset)
    # gr.create_dataset("real_posvel", data=real_posvel, compression="lzf", chunks=True)
    # gr.create_dataset("actions", data=actions, compression="lzf", chunks=True)
    # gr.create_dataset("next_real_posvel", data=next_real_posvel, compression="lzf", chunks=True)
    # gr.create_dataset("next_sim_posvel", data=next_sim_posvel, compression="lzf", chunks=True)
    gr.create_dataset("real_posvel", data=real_posvel, chunks=True)
    gr.create_dataset("actions", data=actions, chunks=True)
    gr.create_dataset("next_real_posvel", data=next_real_posvel, chunks=True)
    gr.create_dataset("next_sim_posvel", data=next_sim_posvel, chunks=True)

    # JUST VERIFYING THAT THE RESHAPING WAS SUCCESSFUL - START


    # m = 1  # for example, because motor i=0 is not moving
    # t = 5  # trajectory
    # plt.plot(x, real_posvel[t, start:end, m], label=f"REAL: motor {m} pos ")
    # plt.plot(
    #     x,
    #     next_sim_posvel[t, start:end, m],
    #     label=f"NEXT_SIM: motor {m} pos",
    #     linestyle="dashed")
    # plt.plot(
    #     x,
    #     next_real_posvel[t, start:end, m],
    #     label=f"NEXT_REAL: motor {m} pos",
    #     linestyle="dashed")
    # plt.plot(
    #     x,
    #     actions[t, start:end, m],
    #     label=f"motor {m} action",
    #     linestyle="dotted")
    #
    # plt.title(f"100Hz real/sim recordings from timestep {start} to {end}")
    # plt.legend()
    # plt.tight_layout()
    #
    # plt.show()
    #
    # # ==================== REAL+SIM VEL - IMPORTANT, SIM VELOCITY IS seemingly INVERTED
    #
    # plt.plot(x, real_posvel[t, start:end, 6 + m], label=f"REAL: motor {m} vel ")
    # plt.plot(
    #     x,
    #     next_sim_posvel[t, start:end, 6 + m],
    #     label=f"NEXT_SIM: motor {m} vel",
    #     linestyle="dashed")
    # plt.plot(
    #     x,
    #     next_real_posvel[t, start:end, 6 + m],
    #     label=f"NEXT_REAL: motor {m} vel",
    #     linestyle="dashed")
    # plt.plot(
    #     x,
    #     actions[t, start:end, m],
    #     label=f"motor {m} action",
    #     linestyle="dotted")
    #
    # plt.title(f"100Hz real/sim recordings from timestep {start} to {end}")
    # plt.legend()
    # plt.tight_layout()
    #
    # plt.show()
    #
    # quit()

    # JUST VERIFYING THAT THE RESHAPING WAS SUCCESSFUL - END














