import os

from numpy.core.fromnumeric import shape
os.environ["NEST_MODULE_PATH"] = "/home/massimo/nest-simulator-2.18.0-build-mpi-music/lib/nest"
os.environ["SLI_PATH"] = "/home/massimo/nest-simulator-2.18.0-build-mpi-music/share/nest/sli"
os.environ["LD_LIBRARY_PATH"] = "/home/massimo/nest-simulator-2.18.0-build-mpi-music/lib/nest:/home/massimo/bin/lib"
os.environ["PATH"] = "/home/massimo/bin/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
os.environ["SPATIALINDEX_C_LIBRARY"] = "/home/massimo/bin/lib/libspatialindex.so"
os.environ["PYTHONPATH"] = "/home/massimo/extra-cereb-nest/Tests:/opt/amber18/lib/python3.6/site-packages/:/home/massimo/.local/nrn/lib/python:"
import sys
sys.path.append('/home/massimo/nest-simulator-2.18.0-build-mpi-music/lib/python3.6/site-packages/')

import mpi4py
import time

import nest
nest.Install("cerebmodule")
nest.Install("util_neurons_module")
import numpy as np
import matplotlib.pyplot as plt

msd = int( time.time() * 1000.0 )
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
nest.SetKernelStatus({'rng_seeds' : range(msd+N_vp+1, msd+2*N_vp+1)})

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

def minimumJerk_ddt(x_init, x_des, T_max, timespan):
    a =  120*(x_des-x_init)/np.power(T_max,5)
    b = -180*(x_des-x_init)/np.power(T_max,4)
    c =  60*(x_des-x_init)/np.power(T_max,3)
    if hasattr(x_init, "shape"):
        d =  np.zeros(x_init.shape)
    else:
        d =  np.zeros(1)

    pol = np.array([a,b,c,d])
    pp  = a*np.power(timespan,3) + b*np.power(timespan,2) + c*np.power(timespan,1) + d

    return pp

time_span = 1000.0
time_vect = np.arange(0, time_span)
buf_size  = 10.0
trial_len = 1200
n_trial   = 1

res = nest.GetKernelStatus('resolution')

Ns = 100 # Number of neurons of sensory feedback and Radial Basis neurons

# Base rates
bas_rate_rb       = 20.0
bas_rate_feedback = 20.0

# Gain
gain_feedback = 1.0

# Rb neuron
gain = 10.0

# Create trajectories
trj_real  = desiredTrajectory(0.0, 100.0, time_span, time_vect)
trj_real  = minimumJerk_ddt(0.0, 1e7,time_span, time_vect)

########## Convert desired and real trajectories into spike trains ##########
##### Sensory feedback
trj_real_p = np.zeros(len(trj_real))
trj_real_p[trj_real>=0] = trj_real[trj_real>=0]
feedback_p = nest.Create("inhomogeneous_poisson_generator",Ns)
for i in feedback_p:
    nest.SetStatus([i], {"rate_times": time_vect[1:], "rate_values": bas_rate_feedback+gain_feedback*np.abs(trj_real_p[1:])})
feedback_p_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(feedback_p,feedback_p_prt, "one_to_one")

trj_real_n = np.zeros(len(trj_real))
trj_real_n[trj_real<0] = trj_real[trj_real<0]
feedback_n = nest.Create("inhomogeneous_poisson_generator",Ns)
for i in feedback_n:
    nest.SetStatus([i], {"rate_times": time_vect[1:], "rate_values": bas_rate_feedback+gain_feedback*np.abs(trj_real_n[1:])})
feedback_n_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(feedback_n,feedback_n_prt, "one_to_one")

############################## Radial Basis neurons ################################
stEst_p = nest.Create("rb_neuron", Ns)
nest.SetStatus(stEst_p, {"kp": gain, "buffer_size":buf_size, "base_rate": bas_rate_rb})
freq_max = 70
signal_sensibility = np.linspace(-freq_max,freq_max,len(stEst_p))
for i,neuron in enumerate(stEst_p):
    nest.SetStatus([neuron], {"desired": signal_sensibility[i]})

syn_exc = {"weight": 1.0}
syn_inh = {"weight": -1.0}

# Connections to RB neurons
nest.Connect(feedback_p_prt, stEst_p, "all_to_all", syn_spec=syn_exc)
nest.Connect(feedback_n_prt, stEst_p, "all_to_all", syn_spec=syn_inh)

########################### DEVICES ###########################
spikedet_feedback_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(feedback_p_prt, spikedet_feedback_p)
spikedet_feedback_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(feedback_n_prt, spikedet_feedback_n)

spikedet_stEst_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(stEst_p, spikedet_stEst_p)

########################### SIMULATION #########################
for trial in range(n_trial):
    nest.Simulate(trial_len)

########################### GET DATA ###########################
SD_feedback_p    = nest.GetStatus(spikedet_feedback_p,keys="events")[0]
SD_feedback_p_ev = SD_feedback_p["senders"]
SD_feedback_p_tm = SD_feedback_p["times"]

SD_feedback_n    = nest.GetStatus(spikedet_feedback_n,keys="events")[0]
SD_feedback_n_ev = SD_feedback_n["senders"]
SD_feedback_n_tm = SD_feedback_n["times"]

SD_stEst_p    = nest.GetStatus(spikedet_stEst_p,keys="events")[0]
SD_stEst_p_ev = SD_stEst_p["senders"]
SD_stEst_p_tm = SD_stEst_p["times"]

########################### PLOTTING ###########################
# Feedback
y_min = np.min(feedback_p_prt)
y_max = np.max(feedback_p_prt)
plt.figure(figsize=(10,8))
plt.scatter(SD_feedback_p_tm , SD_feedback_p_ev-y_min, marker='.', s = 200, label = 'Positive')
plt.scatter(SD_feedback_n_tm , SD_feedback_n_ev-y_max, marker='.', s = 200, label = 'Negative')
plt.title('Scatter plot Feedback', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Neuron ID', size =25)
plt.ylim([0,np.max(feedback_n_prt)-y_max])
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.legend()

x_pos = []
x_neg = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    spikes_pos = [k for k in SD_feedback_p_tm if k<i+delta_t and k>=i]
    spikes_neg = [k for k in SD_feedback_n_tm if k<i+delta_t and k>=i]
    freq_pos = len(spikes_pos)/(delta_t/1000*Ns)
    freq_neg = len(spikes_neg)/(delta_t/1000*Ns)
    x_pos.append(freq_pos)
    x_neg.append(freq_neg)
feedback_signal = [i-j for (i,j) in zip(x_pos, x_neg)]
plt.figure(figsize=(10,8))
t = np.arange(0,trial_len*n_trial,delta_t)
t = np.repeat(t,2)[1:]
x_pos = np.repeat(x_pos,2)[:-1]
plt.plot(t,x_pos)
x_neg = np.repeat(x_neg,2)[:-1]
plt.plot(t,x_neg)
plt.title('Spike frequency Feedback neurons', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
plt.yticks(fontsize=25)

# RB neurons
y_min = np.min(stEst_p)
plt.figure(figsize=(10,8))
plt.scatter(SD_stEst_p_tm , SD_stEst_p_ev-y_min, marker='.', s = 200, label = 'Positive')
plt.title('Scatter plot State estimator', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Neuron ID', size =25)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.legend()

x_pos = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    spikes_pos = [k for k in SD_stEst_p_tm if k<i+delta_t and k>=i]
    freq_pos = len(spikes_pos)/(delta_t/1000*Ns*n_trial)
    x_pos.append(freq_pos)
plt.figure(figsize=(10,8))
t = np.arange(0,trial_len*n_trial,delta_t)
t = np.repeat(t,2)[1:]
x_pos = np.repeat(x_pos,2)[:-1]
plt.plot(t,x_pos)
plt.title('Spike frequency RB neurons', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
plt.yticks(fontsize=25)

plt.show()