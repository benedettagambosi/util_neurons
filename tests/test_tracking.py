import nest
import numpy as np
import matplotlib.pyplot as plt
import time

nest.Install("util_neurons_module")
res = nest.GetKernelStatus("resolution")

# DESIRED TRAJECTORY
def desiredTrajectory(x_init, x_des, T_max, timespan):
    a =   6*(x_des-x_init)/np.power(T_max,5)
    b = -15*(x_des-x_init)/np.power(T_max,4)
    c =  10*(x_des-x_init)/np.power(T_max,3)
    d =  0
    e =  0
    g =  x_init

    pp = a*np.power(timespan,5) + b*np.power(timespan,4) + c*np.power(timespan,3) + g
    #pol = np.array([a,b,c,d,e,g])
    return pp

# Timing
time_span = 1000.0
time_vect = np.arange(0, time_span, res)
buf_size  = 200.0

# Neurons
Nt = 50 # number
bas_rate_track = 30.0 #baseline

# Task definition
pos_init = -20.0
pos_target = 10.0
file_trj_des = "/media/xis/data/work/projects/nest-modules/util_neurons/tests/joint1.dat"

# Generate desired trajectory...
trj_des  = desiredTrajectory(pos_init, pos_target, time_span, time_vect)

# ... save it into the file
a_file = open(file_trj_des, "w")
np.savetxt(a_file, trj_des)
a_file.close()

########################### Create neurons ###########################
track_p = nest.Create("tracking_neuron", Nt)
nest.SetStatus(track_p, {"kp": 10.0, "pos": True, "base_rate": bas_rate_track, "pattern_file": file_trj_des})

track_n = nest.Create("tracking_neuron", Nt)
nest.SetStatus(track_n, {"kp": 10.0, "pos": False, "base_rate": bas_rate_track, "pattern_file": file_trj_des})

########################### DEVICES ###########################
spikedet_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(track_p, spikedet_p)
spikedet_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(track_n, spikedet_n)

########################### SIMULATE ###########################
nest.Simulate(time_span)

########################### GET DATA ###########################
SD_p     = nest.GetStatus(spikedet_p,keys="events")[0]
SD_p_ev  = SD_p["senders"]
SD_p_tm  = SD_p["times"]

SD_n     = nest.GetStatus(spikedet_n,keys="events")[0]
SD_n_ev  = SD_n["senders"]
SD_n_tm  = SD_n["times"]

buf = 10
count_n, bins = np.histogram( SD_n_tm, bins=np.arange(0, time_span+1, buf) )
count_p, bins = np.histogram( SD_p_tm, bins=np.arange(0, time_span+1, buf) )
rate_n = 1000*count_n/(Nt*buf)
rate_p = 1000*count_p/(Nt*buf)

rate_sm_n = np.convolve(rate_n, np.ones(5)/5,mode='same')
rate_sm_p = np.convolve(rate_p, np.ones(5)/5,mode='same')

###################### PLOT ##########################
fig, axs = plt.subplots(5, 1, sharex='col')
axs[0].plot(time_vect,trj_des)
axs[1].plot(SD_p_tm, SD_p_ev,'|')
axs[2].plot(SD_n_tm, SD_n_ev,'|')
axs[3].bar(bins[:-1], rate_p, width=bins[1] - bins[0],color='b')
axs[3].plot(bins[:-1],rate_sm_p,color='k')
axs[4].bar(bins[:-1], rate_n, width=bins[1] - bins[0],color='r')
axs[4].plot(bins[:-1],rate_sm_n,color='k')
plt.show()
