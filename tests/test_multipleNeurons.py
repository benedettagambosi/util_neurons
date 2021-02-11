import nest
import numpy as np
import matplotlib.pyplot as plt
import time

nest.Install("util_neurons_module")

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

time_span = 5000.0
time_vect = np.arange(0, time_span)
buf_size  = 500.0

Ns = 20 # Number of Poisson neurons (source)
Nm = 20 # Number of Motor cortex neurons (destination)

# Base rates
bas_rate_motor   = 5.0;
bas_rate_planner = 5.0;
bas_rate_state   = 5.0;

# Gain
gain_plan  = 15
gain_stEst = 15

# Create trajectories
trj_des  = desiredTrajectory(0.0, 10.0, time_span, time_vect)
trj_err  = -2*np.sin(2*np.pi*(1/time_span)*time_vect)
trj_real = trj_des + trj_err

########## Convert desired and real trajectories into spike trains ##########

##### DESIRED (plan)
trj_des_p = np.zeros(trj_des.size)
trj_des_p[trj_des>=0] = trj_des[trj_des>=0]
plan_p = nest.Create("inhomogeneous_poisson_generator",Ns)
nest.SetStatus(plan_p, {"rate_times": time_vect[1:], "rate_values": bas_rate_planner+gain_plan*np.abs(trj_des_p[1:])})
plan_p_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(plan_p,plan_p_prt, "one_to_one")

trj_des_n = np.zeros(trj_des.size)
trj_des_n[trj_des<0] = trj_des[trj_des<0]
plan_n = nest.Create("inhomogeneous_poisson_generator",Ns)
nest.SetStatus(plan_n, {"rate_times": time_vect[1:], "rate_values": bas_rate_planner+gain_plan*np.abs(trj_des_n[1:])})
plan_n_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(plan_n,plan_n_prt, "one_to_one")

##### REAL (estimator)
trj_real_p = np.zeros(trj_real.size)
trj_real_p[trj_real>=0] = trj_real[trj_real>=0]
stEst_p = nest.Create("inhomogeneous_poisson_generator",Ns)
nest.SetStatus(stEst_p, {"rate_times": time_vect[1:], "rate_values": bas_rate_state+gain_stEst*np.abs(trj_real_p[1:])})
stEst_p_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(stEst_p,stEst_p_prt, "one_to_one")

trj_real_n = np.zeros(trj_real.size)
trj_real_n[trj_real<0] = trj_real[trj_real<0]
stEst_n = nest.Create("inhomogeneous_poisson_generator",Ns)
nest.SetStatus(stEst_n, {"rate_times": time_vect[1:], "rate_values": bas_rate_state+gain_stEst*np.abs(trj_real_n[1:]) })
stEst_n_prt = nest.Create("parrot_neuron",Ns)
nest.Connect(stEst_n,stEst_n_prt, "one_to_one")


############################## Motor cortex ################################
mc_p = nest.Create("basic_neuron", Nm)
nest.SetStatus(mc_p, {"kp": 1.0, "pos": True, "buffer_size":buf_size, "base_rate": bas_rate_motor})

mc_n = nest.Create("basic_neuron", Nm)
nest.SetStatus(mc_n, {"kp": 1.0, "pos": False, "buffer_size":buf_size, "base_rate": bas_rate_motor})

syn_exc = {"weight": 1.0}  # Synaptic weight of the excitatory synapse
syn_inh = {"weight": -1.0} # Synaptic weight of the inhibitory synapse

# Connections to motor cortex positive neurons (sensitive to positive signals)
nest.Connect(plan_p_prt, mc_p, "one_to_one", syn_spec=syn_exc)  # Output of the planner...
nest.Connect(plan_n_prt, mc_p, "one_to_one", syn_spec=syn_inh)
nest.Connect(stEst_p_prt, mc_p, "one_to_one", syn_spec=syn_inh) #... minus output of state estimator
nest.Connect(stEst_n_prt, mc_p, "one_to_one", syn_spec=syn_exc)

# Connections to motor cortex positive neurons (sensitive to negative signals)
nest.Connect(plan_p_prt, mc_n, "one_to_one", syn_spec=syn_exc)
nest.Connect(plan_n_prt, mc_n, "one_to_one", syn_spec=syn_inh)
nest.Connect(stEst_p_prt, mc_n, "one_to_one", syn_spec=syn_inh) #... minus output of state estimator
nest.Connect(stEst_n_prt, mc_n, "one_to_one", syn_spec=syn_exc)


########################### DEVICES ###########################
spikedet_plan_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(plan_p_prt, spikedet_plan_p)
spikedet_plan_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(plan_n_prt, spikedet_plan_n)

spikedet_stEst_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(stEst_p_prt, spikedet_stEst_p)
spikedet_stEst_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(stEst_n_prt, spikedet_stEst_n)

spikedet_mc_p = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(mc_p, spikedet_mc_p)
spikedet_mc_n = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(mc_n, spikedet_mc_n)


########################### SIMULATE ###########################
nest.Simulate(time_span)


########################### GET DATA ###########################
SD_plan_p     = nest.GetStatus(spikedet_plan_p,keys="events")[0]
SD_plan_p_ev  = SD_plan_p["senders"]
SD_plan_p_tm  = SD_plan_p["times"]

SD_plan_n     = nest.GetStatus(spikedet_plan_n,keys="events")[0]
SD_plan_n_ev  = SD_plan_n["senders"]
SD_plan_n_tm  = SD_plan_n["times"]

SD_stEst_p    = nest.GetStatus(spikedet_stEst_p,keys="events")[0]
SD_stEst_p_ev = SD_stEst_p["senders"]
SD_stEst_p_tm = SD_stEst_p["times"]

SD_stEst_n    = nest.GetStatus(spikedet_stEst_n,keys="events")[0]
SD_stEst_n_ev = SD_stEst_n["senders"]
SD_stEst_n_tm = SD_stEst_n["times"]

SD_mc_p    = nest.GetStatus(spikedet_mc_p,keys="events")[0]
SD_mc_p_ev = SD_mc_p["senders"]
SD_mc_p_tm = SD_mc_p["times"]

SD_mc_n    = nest.GetStatus(spikedet_mc_n,keys="events")[0]
SD_mc_n_ev = SD_mc_n["senders"]
SD_mc_n_tm = SD_mc_n["times"]

buf = 100
count_n, bins = np.histogram( SD_mc_n_tm, bins=np.arange(0, time_span+1, buf) )
count_p, bins = np.histogram( SD_mc_p_tm, bins=np.arange(0, time_span+1, buf) )
rate_n = 1000*count_n/(Nm*buf)
rate_p = 1000*count_p/(Nm*buf)

rate_sm_n = np.convolve(rate_n, np.ones(5)/5,mode='same')
rate_sm_p = np.convolve(rate_p, np.ones(5)/5,mode='same')

# plt.figure()
# plt.bar(bins[:-1], rate_n, width=bins[1] - bins[0])
# plt.plot(bins[:-1],rate_sm_n,color='k')
# #plt.bar(bins[:-1], rate_p, width=bins[1] - bins[0])
# plt.show()

# rt_n  = np.zeros(len(time_vect))
# rt_p  = np.zeros(len(time_vect))
# for x in range(int(buf/2),int(time_span-buf/2)):
#     st_idx = x-int(buf/2)
#     ed_idx = x+int(buf/2)
#     rt_n[x] = 1000*len( SD_mc_n_tm[(SD_mc_n_tm>=st_idx) & (SD_mc_n_tm<ed_idx)] )/buf
#     rt_p[x] = 1000*len( SD_mc_p_tm[(SD_mc_p_tm>=st_idx) & (SD_mc_p_tm<ed_idx)] )/buf


########################### PLOTTING ###########################

# Plot trajectories
fig, axs = plt.subplots(3, 1, sharex='col')
ax1, ax2, ax3 = axs
ax1.plot(time_vect,trj_des)
ax1.set_ylabel('desired')
ax2.plot(time_vect,trj_err)
ax2.set_ylabel('error')
ax3.plot(time_vect,trj_des)
ax3.plot(time_vect,trj_real)
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('desired, real')
ax3.legend(['desired','real'])
#plt.show()

fig, axs = plt.subplots(2, 2, sharex='col')
axs[0,0].plot(time_vect, trj_des_p)
axs[1,0].plot(SD_plan_p_tm, SD_plan_p_ev,'|')
axs[0,1].plot(time_vect, trj_des_n)
axs[1,1].plot(SD_plan_n_tm, SD_plan_n_ev,'|')
#plt.show()

fig, axs = plt.subplots(2, 2, sharex='col')
axs[0,0].plot(time_vect, trj_real_p)
axs[1,0].plot(SD_stEst_p_tm, SD_stEst_p_ev,'|')
axs[0,1].plot(time_vect, trj_real_n)
axs[1,1].plot(SD_stEst_n_tm, SD_stEst_n_ev,'|')
#plt.show()

fig, axs = plt.subplots(5, 1, sharex='col')
axs[0].plot(time_vect,trj_des)
axs[0].plot(time_vect,trj_real)
axs[1].plot(SD_mc_p_tm, SD_mc_p_ev,'|')
axs[2].plot(SD_mc_n_tm, SD_mc_n_ev,'|')
axs[3].bar(bins[:-1], rate_n, width=bins[1] - bins[0],color='b')
axs[3].plot(bins[:-1],rate_sm_n,color='k')
axs[4].bar(bins[:-1], rate_p, width=bins[1] - bins[0],color='r')
axs[4].plot(bins[:-1],rate_sm_p,color='k')
#axs[3].plot(time_vect,rt_p)
#axs[3].plot(time_vect,rt_n)
plt.show()
