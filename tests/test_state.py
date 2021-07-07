import os
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

time_span = 1000.0
time_vect = np.arange(0, time_span)
buf_size  = 30.0 #20.0
trial_len = 1200
n_trial   = 1

res = nest.GetKernelStatus('resolution')

Ns = 100 # Number of neurons of sensory feedback and state estimator
Nm = 50  # Number of neurons of cerebellum

# Base rates
bas_rate_stEst   = 0.0
bas_rate_cerebellum = 40.0
bas_rate_feedback   = 40.0

# Gain
gain_cerebellum  = 1.0
gain_feedback = 1.0

# State neuron
gain = 1.0

# Create trajectories
trj_des  = desiredTrajectory(0.0, 100.0, time_span, time_vect)
trj_err  = -150*np.sin(2*np.pi*(1/time_span)*time_vect)
trj_real = trj_des + trj_err
#trj_real  = desiredTrajectory(0.0, 100.0, time_span, time_vect)
#trj_real = trj_des

########## Convert desired and real trajectories into spike trains ##########

##### Cerebellar prediction
trj_des_p = np.zeros(trj_des.size)
trj_des_p[trj_des>=0] = trj_des[trj_des>=0]
cerebellum_p = nest.Create("inhomogeneous_poisson_generator",Nm)
nest.SetStatus(cerebellum_p, {"rate_times": time_vect[1:], "rate_values": bas_rate_cerebellum+gain_cerebellum*np.abs(trj_des_p[1:])+ np.random.normal(0,0*(bas_rate_cerebellum + np.mean(trj_des_p[1:])))})
cerebellum_p_prt = nest.Create("parrot_neuron",Nm)
nest.Connect(cerebellum_p,cerebellum_p_prt, "one_to_one")

trj_des_n = np.zeros(trj_des.size)
trj_des_n[trj_des<0] = trj_des[trj_des<0]
cerebellum_n = nest.Create("inhomogeneous_poisson_generator",Nm)
nest.SetStatus(cerebellum_n, {"rate_times": time_vect[1:], "rate_values": bas_rate_cerebellum+gain_cerebellum*np.abs(trj_des_n[1:])+ np.random.normal(0,0*(bas_rate_cerebellum + np.mean(trj_des_p[1:])))})
cerebellum_n_prt = nest.Create("parrot_neuron",Nm)
nest.Connect(cerebellum_n,cerebellum_n_prt, "one_to_one")

##### Sensory feedback
trj_real_p = np.zeros(trj_real.size)
trj_real_p[trj_real>=0] = trj_real[trj_real>=0]
feedback_p = nest.Create("inhomogeneous_poisson_generator",Ns)
for i in feedback_p:
    nest.SetStatus([i], {"rate_times": time_vect[1:], "rate_values": bas_rate_feedback+gain_feedback*np.abs(trj_real_p[1:])})
feedback_p_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(feedback_p,feedback_p_prt, "one_to_one")

trj_real_n = np.zeros(trj_real.size)
trj_real_n[trj_real<0] = trj_real[trj_real<0]
feedback_n = nest.Create("inhomogeneous_poisson_generator",Ns)
for i in feedback_n:
    nest.SetStatus([i], {"rate_times": time_vect[1:], "rate_values": bas_rate_feedback+gain_feedback*np.abs(trj_real_n[1:])})
feedback_n_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(feedback_n,feedback_n_prt, "one_to_one")

############################## State estimator ################################
stEst_p = nest.Create("state_neuron", Ns)
nest.SetStatus(stEst_p, {"kp": gain, "buffer_size":buf_size, "base_rate": bas_rate_stEst})

stEst_n = nest.Create("state_neuron", Ns)
nest.SetStatus(stEst_n, {"kp": gain, "buffer_size":buf_size, "base_rate": bas_rate_stEst})

syn_exc_1 = {"weight": 1.0, "receptor_type": 1}
syn_inh_1 = {"weight": -1.0, "receptor_type": 1}
syn_exc_2 = {"weight": 1.0, "receptor_type": 2}
syn_inh_2 = {"weight": -1.0, "receptor_type": 2}

# Connections to state estimator positive neurons
nest.Connect(cerebellum_p_prt, stEst_p, "all_to_all", syn_spec=syn_exc_1)
nest.Connect(feedback_p_prt, stEst_p, "all_to_all", syn_spec=syn_inh_2) 
nest.SetStatus(stEst_p, {"num_first": float(len(cerebellum_p_prt)), "num_second":float(len(feedback_p_prt))})

# Connections to state estimator negative neurons
nest.Connect(cerebellum_n_prt, stEst_n, "all_to_all", syn_spec=syn_inh_1)
nest.Connect(feedback_n_prt, stEst_n, "all_to_all", syn_spec=syn_exc_2)
nest.SetStatus(stEst_n, {"num_first": float(len(cerebellum_n_prt)), "num_second":float(len(feedback_n_prt))})

########################### DEVICES ###########################
spikedet_cerebellum_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(cerebellum_p_prt, spikedet_cerebellum_p)
spikedet_cerebellum_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(cerebellum_n_prt, spikedet_cerebellum_n)

spikedet_feedback_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(feedback_p_prt, spikedet_feedback_p)
spikedet_feedback_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(feedback_n_prt, spikedet_feedback_n)

spikedet_stEst_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(stEst_p, spikedet_stEst_p)
spikedet_stEst_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(stEst_n, spikedet_stEst_n)

t = time.time()
########################### SIMULATION #########################
for trial in range(n_trial):
    nest.Simulate(trial_len)
print('Elapsed time (s): ', time.time()-t)
########################### GET DATA ###########################
SD_cerebellum_p     = nest.GetStatus(spikedet_cerebellum_p,keys="events")[0]
SD_cerebellum_p_ev  = SD_cerebellum_p["senders"]
SD_cerebellum_p_tm  = SD_cerebellum_p["times"]

SD_cerebellum_n     = nest.GetStatus(spikedet_cerebellum_n,keys="events")[0]
SD_cerebellum_n_ev  = SD_cerebellum_n["senders"]
SD_cerebellum_n_tm  = SD_cerebellum_n["times"]

SD_feedback_p    = nest.GetStatus(spikedet_feedback_p,keys="events")[0]
SD_feedback_p_ev = SD_feedback_p["senders"]
SD_feedback_p_tm = SD_feedback_p["times"]

SD_feedback_n    = nest.GetStatus(spikedet_feedback_n,keys="events")[0]
SD_feedback_n_ev = SD_feedback_n["senders"]
SD_feedback_n_tm = SD_feedback_n["times"]

SD_stEst_p    = nest.GetStatus(spikedet_stEst_p,keys="events")[0]
SD_stEst_p_ev = SD_stEst_p["senders"]
SD_stEst_p_tm = SD_stEst_p["times"]

SD_stEst_n    = nest.GetStatus(spikedet_stEst_n,keys="events")[0]
SD_stEst_n_ev = SD_stEst_n["senders"]
SD_stEst_n_tm = SD_stEst_n["times"]

########################### PLOTTING ###########################
# cerebellum
y_min = np.min(cerebellum_p_prt)
y_max = np.max(cerebellum_p_prt)
plt.figure(figsize=(10,8))
plt.scatter(SD_cerebellum_p_tm , SD_cerebellum_p_ev-y_min, marker='.', s = 200, label = 'Positive')
plt.scatter(SD_cerebellum_n_tm , SD_cerebellum_n_ev-y_max, marker='.', s = 200, label = 'Negative')
plt.title('Scatter plot cerebellum', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Neuron ID', size =25)
plt.ylim([0,np.max(cerebellum_n_prt)-y_max])
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.legend()

x_pos = []
x_neg = []
delta_t = 5
for i in range(0,trial_len*n_trial,delta_t):
    spikes_pos = [k for k in SD_cerebellum_p_tm if k<i+delta_t and k>=i]
    spikes_neg = [k for k in SD_cerebellum_n_tm if k<i+delta_t and k>=i]
    freq_pos = len(spikes_pos)/(delta_t/1000*Nm)
    freq_neg = len(spikes_neg)/(delta_t/1000*Nm)
    x_pos.append(freq_pos)
    x_neg.append(freq_neg)
cerebellar_signal = [i-j for (i,j) in zip(x_pos, x_neg)]
plt.figure(figsize=(10,8))
t = np.arange(0,trial_len*n_trial,delta_t)
t = np.repeat(t,2)[1:]
x_pos = np.repeat(x_pos,2)[:-1]
plt.plot(t,x_pos)
x_neg = np.repeat(x_neg,2)[:-1]
plt.plot(t,x_neg)
plt.title('Spike frequency cerebellum neurons', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
plt.yticks(fontsize=25)

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
delta_t = 5
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

# State estimator
y_min = np.min(stEst_p)
plt.figure(figsize=(10,8))
plt.scatter(SD_stEst_p_tm , SD_stEst_p_ev-y_min, marker='.', s = 200, label = 'Positive')
plt.scatter(SD_stEst_n_tm , SD_stEst_n_ev-y_min, marker='.', s = 200, label = 'Negative')
plt.title('Scatter plot State estimator', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Neuron ID', size =25)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.legend()

x_pos = []
x_neg = []
delta_t = 5
for i in range(0,trial_len*n_trial,delta_t):
    spikes_pos = [k for k in SD_stEst_p_tm if k<i+delta_t and k>=i]
    spikes_neg = [k for k in SD_stEst_n_tm if k<i+delta_t and k>=i]
    freq_pos = len(spikes_pos)/(delta_t/1000*Ns*n_trial)
    freq_neg = len(spikes_neg)/(delta_t/1000*Ns*n_trial)
    x_pos.append(freq_pos)
    x_neg.append(freq_neg)
state_signal = [i-j for (i,j) in zip(x_pos, x_neg)]
plt.figure(figsize=(10,8))
t = np.arange(0,trial_len*n_trial,delta_t)
t = np.repeat(t,2)[1:]
x_pos = np.repeat(x_pos,2)[:-1]
plt.plot(t,x_pos)
x_neg = np.repeat(x_neg,2)[:-1]
plt.plot(t,x_neg)
plt.title('Spike frequency State estimator neurons', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
plt.yticks(fontsize=25)

plt.figure(figsize=(10,8))
plt.plot(np.arange(0,trial_len*n_trial,delta_t),cerebellar_signal, label='Cerebellum')
plt.plot(np.arange(0,trial_len*n_trial,delta_t),feedback_signal, label='Feedback')
plt.plot(np.arange(0,trial_len*n_trial,delta_t),state_signal, label='State')
plt.title('Bayesian integration', size =25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.legend()
figure_folder = './'
plt.savefig(figure_folder+'Bayesian integration.svg')

def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

total_time = n_trial*trial_len
# Data stored every 50 ms for the 200 neurons
n_tot_lines = total_time/50*200
step_size = 200

file = open('/home/massimo/Scrivania/dottorato/bsb_env/variability.txt', 'r')
Lines = file.readlines()
count = 0
times = []
cerebellar_variability = []
feedback_variability = []
for line in Lines:
    count += 1
    splitted = line.strip().split(':')
    [time, values] = splitted[1].split(';')
    [cerebellum, feedback] = values.split(',')
    if count % step_size == 0:
        times.append(float(time))
        cerebellar_variability.append(float(cerebellum))
        feedback_variability.append(float(feedback))

plt.figure()
plt.plot(times, cerebellar_variability, label = 'cerebellum')
plt.plot(times, feedback_variability, label = 'feedback')
plt.legend()
plt.savefig(figure_folder+'Variability plot.svg')

plt.show()