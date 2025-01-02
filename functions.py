from neuron import h
from neuron.units import ms, mV, um
h.load_file("stdrun.hoc")

import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from scipy.stats import pearsonr
from scipy.stats import lognorm


# Single Neuron 
class Cell:
    def __init__(self,
                 type : str,
                 n_idx : int):
        """
        Parameters
        ----------
        type : Cell type, Exciatory or Inhibitory
        n_id : Neuron idx
        """
        self.type = type
        self.n_idx = n_idx
        
        self._setup_morphology_biophysics()
        self._setup_spike_detector()

    def _setup_morphology_biophysics(self):
        self.soma = h.Section(name="soma", cell=self)
        self.soma.L = self.soma.diam = 12.6157 * um     # Set the L, diameter of a cylindrical soma 
        self.soma.Ra = 100                              # Axial resistance in Ohm * cm 
        self.soma.cm = 1                                # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')                          # 'Hodgkin-Huxley' model 
        for seg in self.soma:
            seg.hh.gnabar = 0.12                        # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036                        # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003                          # Leak conductance in S/cm2
            seg.hh.el = -54.3

    def _setup_spike_detector(self):
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)      # Record membrane voltage
        self.n_ap = h.APCount(self.soma(0.5))                       # Count of action potential 
        
        # Record spike times
        self.spike_times = h.Vector() 
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self._spike_detector.record(self.spike_times)

    def __repr__(self) -> str:
        return f'{self.type}[{self.n_idx}]'


# Network of Neurons
class Network:
    def __init__(self,
                 n : int,
                 ratio : float,
                 corr : str,
                 mean : int, sd : int,
                 seed = 42, 
                 connection = True,
                 scale = 1):
        '''
        Parameters
        ----------
        n : Number of cells in network
        ratio : Ratio of E cell and I cell
        corr : Correlation type of network [Random, Pre, Post, PrePost]
        mean : Mean of log-normal distribution
        sd : Standard deviation of log-normal distribution
        seed : Seed for reproducibility, Default : 42
        connection : Whether make connection between cells, Default : True
        scale : Adjust scale of synaptic weight matrix, Default : 1
        '''
        
        self.n = n                                      # Number of cell in network
        self.ratio = ratio                              # Ratio of E cell and I cell 

        self.corr = corr                                # Correlation type of network (Random, Pre, Post, PrePost)
        self.mean = mean                                # Mean of log-normal distribution 
        self.sd = sd                                    # Standard deviation of log-normal distribution
        self.weight_matrix = np.zeros((self.n, self.n)) # Initialize weight matrix

        self.seed = seed                                # Seed for reproducibility
        self.connection = connection                    # Whether make connection between cells / Defalut : True
        self.scale = scale                              # Adjust scale of synaptic weight matrix / ex) scale = 10 makes X10 of weight matrix 
        self.add_time = 0                               # Additional time for input

        # Cell container
        self.E_cells = [] 
        self.I_cells = []
        self.total_cells = []
    
        # Synapses and NetCons container
        self.syns = [] 
        self.ncs = [] 
        self.weight_matrix = np.zeros((self.n, self.n))
        
        self._create_cells()
        if self.connection:
            self._create_matrix()
            self._create_network()

    def _create_cells(self):
        self.n_E = int(self.n * self.ratio)             # Number of E cell
        self.n_I = self.n - self.n_E                    # Number of I cell
        
        self.E_cells = [Cell('E', i) for i in range(self.n_E)]
        self.I_cells = [Cell('I', i) for i in range(self.n_E, self.n)]
        self.total_cells = self.E_cells + self.I_cells

    def _create_matrix(self):
        if self.seed is not None:
            np.random.seed(self.seed) 

        # Extract weight matrix from log-normal distribution
        # By the correlation type, adjust the weight matrix with correlation factors 

        assert self.corr in ['Random', 'Pre', 'Post', 'PrePost'], "Please check the correlation type"
      
        if self.corr == 'Random':
            self.weight_matrix = np.random.lognormal(self.mean, self.sd, size=(self.n, self.n))
        
        elif self.corr == 'Pre':
            self.weight_matrix = np.random.lognormal(self.mean, self.sd, size=(self.n, self.n))
            correlation_factors = np.random.lognormal(self.mean, self.sd, size=self.n)
            self.weight_matrix = self.weight_matrix.T * correlation_factors

        elif self.corr == 'Post':
            self.weight_matrix = np.random.lognormal(self.mean, self.sd, size=(self.n, self.n))
            correlation_factors = np.random.lognormal(self.mean, self.sd, size=self.n)
            self.weight_matrix = (self.weight_matrix.T * correlation_factors).T

        elif self.corr == 'PrePost':
            self.weight_matrix = np.random.lognormal(self.mean, self.sd, size=(self.n, self.n))

            pre_correlation_factors = np.random.lognormal(self.mean, self.sd, size=self.n)
            self.weight_matrix = self.lognormal_matrix.T * pre_correlation_factors

            post_correlation_factors = np.random.lognormal(self.mean, self.sd, size=self.n)
            self.weight_matrix = (self.weight_matrix.T * post_correlation_factors).T

        # Balancing the weight of E and I cells
        sum_E = np.sum(self.weight_matrix[:, :self.n_E])
        sum_I = np.sum(self.weight_matrix[:, self.n_E:])

        if sum_E == 0 or sum_I == 0:
            raise ValueError("Please check the weight matrix")
        
        scaling_factor = (sum_I / 6.5) / sum_E
        self.weight_matrix[:, :self.n_E] *= scaling_factor

        # Adjust the scale of weight matrix (Optional)
        assert self.scale > 0, "Please check the scale factor"

        self.weight_matrix *= self.scale
        
    def _create_network(self):
        for i in range(self.n): 
            for j in range(self.n):
                if i == j:
                    continue

                pre, post = self.total_cells[j], self.total_cells[i]
                syn = h.ExpSyn(post.soma(0.5))
                nc = h.NetCon(pre.soma(0.5)._ref_v, syn, sec=pre.soma)
                nc.delay = 2.5 * ms 

                if pre.type == 'E':
                    syn.e = 0 * mV              # Reversal potential of excitatory synapse
                    syn.tau = 0.1 * ms          # Time constant of the synapse
                    nc.weight[0] = self.weight_matrix[i, j]

                elif pre.type == 'I':
                    syn.e = -75 * mV            # Reversal potential of inhibitory synapse
                    syn.tau = 0.1 * ms          # Time constant of the synapse
                    nc.weight[0] = self.weight_matrix[i, j]
                
                else:
                    raise ValueError("Please check the type of cell")
                
                # Should store synapses and NetCons
                self.syns.append(syn)
                self.ncs.append(nc)

    def __iter__(self):
        return iter(self.total_cells)
    
    # Plot
    def plot_weight_matrix(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(self.weight_matrix, ax=ax, cmap='jet')

        ax.set_xticks([0, self.n_E])
        ax.set_xticklabels(['E', 'I'])
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        ax.set_yticks([0, self.n_E])
        ax.set_yticklabels(['E', 'I'], rotation=0, ha='right')
        ax.tick_params(axis='both', which='major')

        ax.set_xlabel('Presynaptic Cell')
        ax.set_ylabel('Postsynaptic Cell')
        ax.set_title('Weight Matrix')

        plt.show()
    
    def plot_weight_distribution(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        flattened = self.weight_matrix.flatten()
        mean = np.mean(flattened)
        sd = np.std(flattened)

        ax.hist(flattened)
        ax.text(0.8, 0.8, f'Mean: {mean:.5f}\n SD: {sd:.5f}', transform=ax.transAxes)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.set_xlabel('Synaptic Weight')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Log(Weight)')

        plt.show()


class Poisson_Input:
    def __init__(self, rate = 200):
        self.rate = rate
        self._setup_noise()

    # Current setup is for subthreshold dynamics 
    # Further modification is needed to generate 4~6 Hz of firing rate (Implemented in the Simulation Class)
    def _setup_noise(self):
        self.netstim = h.NetStim()
        self.netstim.interval = 1000 / self.rate         # Interval between spikes (ms)
        self.netstim.number = 1e9                   # Number of spikes
        self.netstim.start = 0                      # Start time of the stimulus
        self.netstim.noise = 0.8                    # Randomness Range : (0,1) 0 is randomless 


class Simulation:
    def __init__(self,
                 networks: list, 
                 run_time: int, 
                 poisson = True, 
                 seed = 42):
        '''
        Parameters
        ----------
        networks : List of networks
        run_time : Simulation time
        poisson : Whether use poisson input or not
        seed : Seed for reproducibility, Default : 42
        '''

        self.networks = networks
        self.n = networks[0].n  
        self.n_E = int(self.n * networks[0].ratio)
        self.results = [{} for _ in networks]       # Result container

        self.run_time = run_time
        self.poisson = poisson 
        self._t = None                              # Time vector for simulation
        self.seed = seed 

        # Containers for IClamp, NetStim, NetCon, and Synapse
        self.ICs = []
        self.nss = []
        self.ncs = []
        self.syns = []
        
        if self.poisson:
            self._create_poisson()

    # Poisson Input
    def _create_poisson(self):
        if self.seed is not None:
            random.seed(42)

        # Same poisson input for each network ; Different neuron index gets different poisson input
        '''
            In the Poisson_Input class, we generate only subthreshold dynamics. 
            To achieve a firing rate within a specified range, we randomly select spike times and use IClamp to generate spikes at those times.
            The firing rate and noise can be controlled by adjusting the rate and noise parameters of the Poisson_Input class, which is typically the most straightforward approach. 
            Without relying on specific spike times or the IClamp method, narrowing the firing rate range becomes more complex.
            To adjust the firing rate range, modify the range of random.uniform(), for example: int(random.uniform(0.004, 0.006) * self.run_time)
        ''' 
        if self.poisson:
            assert self.run_time > 0, "Please check the simulation time"
            times = [random.sample(range(0, self.run_time), int(random.uniform(0.004, 0.006)*self.run_time)) for _ in range(self.n)]

            poisson_inputs = [Poisson_Input(200) for _ in range(self.n)]

            for i, cells in enumerate(zip(*self.networks)):
                for cell in cells:
                    syn = h.ExpSyn(cell.soma(0.5))
                    nc = h.NetCon(poisson_inputs[i].netstim, syn)
                    nc.weight[0] = 0.002
                    
                    time = times[i]
                    for j in time:
                        ic = h.IClamp(cell.soma(0.5))
                        ic.delay = j
                        ic.amp = 0.1
                        ic.dur = 1 * ms
                        self.ICs.append(ic)

                    self.syns.append(syn)
                    self.ncs.append(nc)

    # Additional Input 
    def add_input(self,
                  add_input_nets = [],
                  add_prop = 0.2,
                  input_param = (1, 0, 0, 0),
                  input_mode = 'E'):
        '''
        Parameters
        ----------
        add_input_nets : List of networks that get additional input
        add_prop : Proportion of cells to get additional input 
        input_param : Tuple of input parameters (start, noise, interval, number)
        input_mode : To which type of cells get additional input, Default : 'E' (Only E cells)
        '''

        if self.seed is not None:
            random.seed(42)

        self.add_input_nets = add_input_nets 
        self.add_prop = add_prop 
        self.input_mode = input_mode 
        start, noise, interval, number = input_param  
        
        if self.add_input_nets:
            if input_mode == 'E':           # Input only to E cells
                self.selected = random.sample(range(self.n_E), int(self.add_prop * self.n))
            elif input_mode == 'I':         # Input only to I cells
                self.selected = random.sample(range(self.n_E, self.n), int(self.add_prop * self.n))
            else:                           # Input to all cells
                self.selected = random.sample(range(self.n), int(self.add_prop * self.n))

            for i in self.selected:
                synapses = []
                stimulators = []
                connections = []

                for network in self.add_input_nets:
                    c = network.total_cells[i]
                    syn = h.ExpSyn(c.soma(0.5))
                    stim = h.NetStim()

                    # Stimulus parameters
                    stim.start = start
                    stim.noise = noise
                    stim.interval = interval
                    stim.number = number

                    nc = h.NetCon(stim, syn)
                    nc.weight[0] = 0.01

                    synapses.append(syn)
                    stimulators.append(stim)
                    connections.append(nc)

                self.ncs.append(connections)
                self.nss.append(stimulators)
                self.syns.append(synapses)
        
        add_time = interval * number
        for network in add_input_nets:
            network.add_time = add_time

    def run(self):
        self._t = h.Vector().record(h._ref_t)

        h.finitialize(-65 * mV)
        h.continuerun(self.run_time * ms)

        for idx, network in enumerate(self.networks): 
            for i, n in enumerate(network.total_cells):
                self.results[idx][f'Neuron[{i}]'] = {
                    'AP': int(n.n_ap.n),
                    'voltage': n.soma_v,
                    'spike': n.spike_times.to_python()
                }

        print("Simulation complete")

    def raster_plot(self, net_idx : int):

        assert net_idx in range(len(self.networks)), "Please check the network index"

        fig, ax = plt.subplots(figsize=(5, 5))
        total_cell = self.networks[net_idx].total_cells
        for i, cell in enumerate(total_cell):
            if i < self.n_E:
                ax.vlines(cell.spike_times.to_python(), i-0.1, i+0.1, color='b')
            else:
                ax.vlines(cell.spike_times.to_python(), i-0.1, i+0.1, color='r')
        ax.set_yticks([self.n_E - 1, self.n - 1])
        ax.set_yticklabels(['E','I'])
        ax.set_xlim([0, self.run_time])
        ax.set_ylabel('Neurons')
        ax.set_xlabel('t (ms)')
        ax.set_title(f'N={self.n}, run_time={self.run_time} ms\n {self.networks[net_idx].corr} Corr Network')

        plt.show()

    def firing_rate(self, net_idx : int):

        assert net_idx in range(len(self.networks)), "Please check the network index"
        assert self.results, "Please run the simulation first"

        fig, ax = plt.subplots(figsize=(5, 5))
        add_time = self.networks[net_idx].add_time

        spike_times = [v['spike'] for v in self.results[net_idx].values()]
        spike_times = [[element for element in sublist if element > add_time] for sublist in spike_times]
        firing_rate = [(len(sublist) / (self.run_time - add_time) * 1000) for sublist in spike_times]

        fr_mean = np.mean(firing_rate)
        fr_std = np.std(firing_rate)

        shape, loc, scale = lognorm.fit(firing_rate, floc=0)
        x = np.linspace(min(firing_rate), max(firing_rate), 1000)
        p = lognorm.pdf(x, s=shape, scale=scale)

        ax.hist(firing_rate, density=True, alpha=1)
        ax.plot(x, p, 'k', linewidth=1, alpha=0.8)
        ax.text(0.6, 0.8, f'Mean: {fr_mean:.5f}\n SD: {fr_std:.5f}', transform=ax.transAxes)
        ax.set_xlabel('Firing Rates (sp/s)')
        ax.set_ylabel('Density')
        ax.set_title(f'N={self.n}, run_time={self.run_time} ms\n {self.networks[net_idx].corr} Corr Network')
        
        plt.show()

    def fr_scatter(self, net_idx_1 : int, net_idx_2 : int):
        assert net_idx_1 in range(len(self.networks)), "Please check the network index"
        assert net_idx_2 in range(len(self.networks)), "Please check the network index"
        assert self.results, "Please run the simulation first"

        fig, ax = plt.subplots(figsize=(5, 5))

        n1 = self.networks[net_idx_1]
        n2 = self.networks[net_idx_2]

        add_time_1 = n1.add_time
        spike_times_1 = [v['spike'] for v in self.results[net_idx_1].values()]
        spike_times_1 = [[element for element in sublist if element > add_time_1] for sublist in spike_times_1]
        firing_rate_1 = [(len(sublist) / (self.run_time - add_time_1) * 1000) for sublist in spike_times_1]

        add_time_2 = n2.add_time
        spike_times_2 = [v['spike'] for v in self.results[net_idx_2].values()]
        spike_times_2 = [[element for element in sublist if element > add_time_2] for sublist in spike_times_2]
        firing_rate_2 = [(len(sublist) / (self.run_time - add_time_2) * 1000) for sublist in spike_times_2]
        
        max_value = np.max([np.max(firing_rate_1), np.max(firing_rate_2)])
        min_value = np.min([np.min(firing_rate_1), np.min(firing_rate_2)])

        ax.scatter(firing_rate_1[:self.n_E], firing_rate_2[:self.n_E], s=25, alpha=0.5, color='b', label='E')
        ax.scatter(firing_rate_1[self.n_E:], firing_rate_2[self.n_E:], s=25, alpha=0.5, color='r', label='I')
        ax.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Firing Rate (Network 1)')
        ax.set_ylabel('Firing Rate (Network 2)')
        ax.grid(True, linestyle='--') 
        ax.set_xlim(0, max_value+1)
        ax.set_ylim(0, max_value+1)
        ax.legend(loc='lower right')
        ax.set_title(f'N={self.n}, run_time={self.run_time} ms\n {self.networks[net_idx_1].corr} Corr Network')

        corr, pval = pearsonr(firing_rate_1, firing_rate_2)
        ax.text(0.05, 0.95, f'r = {corr:.2f}\np = {pval:.2e}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.show()
