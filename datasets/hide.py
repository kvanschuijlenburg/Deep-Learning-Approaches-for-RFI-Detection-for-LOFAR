import os
from datetime import timedelta
from datetime import datetime

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from numpy import ceil, pi, random
from scipy import stats
from scipy.signal.signaltools import fftconvolve
from collections import namedtuple

#from copited from ivy.utils.struct
class ImmutableStruct(object):#, UserDict.DictMixin):
    """
    A `dict`-like object, whose keys can be accessed with the usual
    '[...]' lookup syntax, or with the '.' get attribute syntax.

    Examples::

      >>> a = Struct()
      >>> a['x'] = 1
      >>> a.x
      1
      >>> a.y = 2
      >>> a['y']
      2

    Values can also be initially set by specifying them as keyword
    arguments to the constructor::

      >>> a = Struct(z=3)
      >>> a['z']
      3
      >>> a.z
      3

    Like `dict` instances, `Struct`s have a `copy` method to get a
    shallow copy of the instance:

      >>> b = a.copy()
      >>> b.z
      3

    """
    def __init__(self, initializer=None, **extra_args):
        if initializer is not None:
            try:
                # initializer is `dict`-like?
                for name, value in initializer.items():
                    self.__dict__[name] = value
            except AttributeError:
                # initializer is a sequence of (name,value) pairs?
                for name, value in initializer:
                    self.__dict__[name] = value
        for name, value in extra_args.items():
            self.__dict__[name] = value

    def copy(self):
        """Return a (shallow) copy of this `Struct` instance."""
        return ImmutableStruct(self)

    # the `DictMixin` class defines all std `dict` methods, provided
    # that `__getitem__`, `__setitem__` and `keys` are defined.
    def __setitem__(self, name, val):
        raise Exception("Trying to modify immutable struct with: %s=%s"%(str(name), str(val)))
        
    def __getitem__(self, name):
        return self.__dict__[name]
    
    def keys(self):
        return self.__dict__.keys()
    
    def __str__(self):
        str = "{\n"
        for name, value in self.items():
            str += ("%s='%s'\n" %(name, value))
        str += "}"
        return str
class Struct(ImmutableStruct):
    """
    Mutable implementation of a Strcut
    """
    def __setitem__(self, name, val):
        self.__dict__[name] = val
        
    def copy(self):
        """Return a (shallow) copy of this `Struct` instance."""
        return Struct(self)        
CoordSpec = namedtuple("CoordSpec", ["time", "alt", "az",  "ra", "dec"])

load_rfi_template = False
rfideltat = 5    #1                       # Width in time for RFI [units of pixels]
rfideltaf = .5  #1                        # Width in frequency for RFI [units of pixels]
rfiexponent = 2   #1                      # Exponential model (1) or Gaussian model (2) for RFI
rfienhance = 1.7    #1                    # Enhance fraction covered by RFI
rfiday = (6.0, 22.0)                    # Beginning and end of RFI day
rfidamping = 0.1                        # Damping factor of RFI during the RFI night 

class Plugin_Add_RFI_Phaseswitch():
    """
    Adds RFI to the time ordered data (phase switch).
    
    """
    def __init__(self, ctx) -> None:
        self.ctx=ctx

    def __call__(self):
        params = self.ctx.params
        time = self.getTime()
        freq = self.ctx.frequencies
        if params.load_rfi_template:
            mod = importlib.import_module(self.ctx.params.instrument)
            rfifrac, rfiamplitude = mod.get_rfi_params(self.ctx.frequencies)
            params.rfifrac = rfifrac
            params.rfiamplitude = rfiamplitude
            
        try:
            rfiday = params.rfiday
        except AttributeError:
            rfiday = (0.0, 24.0)

        try:
            rfidamping = params.rfidamping
        except AttributeError:
            rfidamping = 1.0

        rfi = getRFI(params.white_noise_scale, params.rfiamplitude,
                     params.rfifrac, params.rfideltat,
                     params.rfideltaf, params.rfiexponent,
                     params.rfienhance, freq, time, rfiday, rfidamping)
        self.ctx.tod_vx += rfi
        self.ctx.tod_vx_rfi = rfi 
        return rfi

    def getTime(self):
        time = []
        for coord in self.ctx.strategy_coords:
            t = self.ctx.strategy_start + timedelta(seconds=coord.time)
            time.append(t.hour + t.minute / 60. + t.second / 3600.)
        return np.asarray(time)

    def __str__(self):
        return "Add RFI (phase switch)"

def getRFI(background, amplitude, fraction, deltat, deltaf, exponent, enhance,
           frequencies, time, rfiday, damping):
    """
    Get time-frequency plane of RFI.
     
    :param background: background level of data per channel
    :param amplitude: maximal amplitude of RFI per channel
    :param fraction: fraction of RFI dominated pixels per channel
    :param deltat: time scale of rfi decay (in units of pixels)
    :param deltaf: frequency scale of rfi decay (in units of pixels)
    :param exponent: exponent of rfi model (either 1 or 2)
    :param enhance: enhancement factor relative to fraction
    :param frequencies: frequencies of tod in MHz
    :param time: time of day in hours of tod
    :param rfiday: tuple of start and end of RFI day
    :param damping: damping factor for RFI fraction during the RFI night
    :returns RFI: time-frequency plane of RFI 
    """
    assert rfiday[1] >= rfiday[0], "Beginning of RFI day is after it ends."
    r = 1 - (rfiday[1] - rfiday[0]) / 24.
    nf = frequencies.shape[0]
    if (r == 0.0) | (r == 1.0):
        RFI = calcRFI(background, amplitude, fraction,
                      deltat, deltaf, exponent, enhance,
                      nf, time.shape[0])
    else:
        day_night_mask = getDayNightMask(rfiday, time)
        # Get fractions of day and night
        fday = np.minimum(1, fraction * (1 - damping * r)/(1 - r))
        fnight = (fraction - fday * (1 - r)) / r
        nday = day_night_mask.sum()
        nnight = time.shape[0] - nday
        RFI = np.zeros((nf, time.shape[0]))
        if nnight > 0:
            RFI[:,~day_night_mask] = calcRFI(background, amplitude, fnight,
                                             deltat, deltaf, exponent, enhance,
                                             nf, nnight)
        if nday > 0:
            RFI[:,day_night_mask] = calcRFI(background, amplitude, fday,
                                            deltat, deltaf, exponent, enhance,
                                            nf, nday)
    return RFI

def calcRFI(background, amplitude, fraction, deltat, deltaf, exponent, enhance,
           nf, nt):
    """
    Get time-frequency plane of RFI.
     
    :param background: background level of data per channel
    :param amplitude: maximal amplitude of RFI per channel
    :param fraction: fraction of RFI dominated pixels per channel
    :param deltat: time scale of rfi decay (in units of pixels)
    :param deltaf: frequency scale of rfi decay (in units of pixels)
    :param exponent: exponent of rfi model (either 1 or 2)
    :param enhance: enhancement factor relative to fraction
    :param nf: number of frequency channels
    :param nt: number of time steps
    :returns RFI: time-frequency plane of RFI 
    """
    lgb = np.log(background)
    lgA = np.log(amplitude)
    d = lgA - lgb
    # choose size of kernel such that the rfi is roughly an order of magnitude
    # below the background even for the strongest RFI
    Nk = int(ceil(np.amax(d))) + 3
    t = np.arange(nt)
    if exponent == 1:
        n = d * d * (2. * deltaf * deltat / 3.0)
    elif exponent == 2:
        n = d * (deltaf * deltat * pi *.5)
    else:
        raise ValueError('Exponent must be 1 or 2, not %d'%exponent)
    neff = fraction * enhance * nt / n
    N = np.minimum(random.poisson(neff, nf), nt)
    RFI = np.zeros((nf,nt))
    dt = int(ceil(.5 * deltat))
    # the negative indices really are a hack right now
    neginds = []
    for i in range(nf):
#         trfi = choice(t, N[i], replace = False)
        trfi = random.permutation(t)[:N[i]]
#         trfi = randint(0,nt,N[i])
        r = random.rand(N[i])
        tA = np.exp(r * d[i] + lgb[i])
        r = np.where(random.rand(N[i]) > .5, 1, -1)
        sinds = []
        for j in range(dt):
            fac = (-1)**j * (j + 1) * dt
            sinds.append(((trfi + fac * r) % nt))
        neginds.append(np.concatenate(sinds))
        RFI[i,trfi] = tA
    k = kernel(deltaf, deltat, nf, nt, Nk, exponent)
    RFI = fftconvolve(RFI, k, mode = 'same')
#     neginds = np.unique(concatenate(neginds))
#     RFI[:,neginds] *= -1
    df = int(ceil(deltaf))
    for i, idxs in enumerate(neginds):
        mif = np.maximum(0, i-df)
        maf = np.minimum(nf, i+df)
        RFI[mif:maf,idxs] *= -1
    return RFI

def getDayNightMask(rfiday, time):
    return (rfiday[0] < time) & (time < rfiday[1])

def logmodel(x, dx, exponent):
    """
    Model for the log of the RFI profile:
     * -abs(x)/dx for exponent 1
     * -(x/dx)^2 for exponent 2

    :param x: grid on which to evaluate the profile
    :param dx: width of exponential
    :param exponent: exponent of (x/dx), either 1 or 2
    :returns logmodel: log of RFI profile
    """
    if exponent == 1:
        return -np.absolute(x)/dx
    elif exponent == 2:
        return -(x * x) / (dx * dx)
    else:
        raise ValueError('Exponent must be 1 or 2, not %d'%exponent)
    
def kernel(deltaf, deltat, nf, nt, N, exponent):
    """
    Convolution kernel for FFT convolution
    
    :param deltaf: spread of RFI model in frequency
    :param deltat: spread of RFI model in time
    :param nf: number of frequencies
    :param nt: number of time steps
    :param N: size of kernel relative to deltaf, deltat
    :param exponent: exponent of RFI model (see logmodel)
    :returns kernel: convolution kernel
    """
    fmax, tmax = np.minimum([N * deltaf, N * deltat], [(nf-1)/2,(nt-1)/2])
    f = np.arange(2*fmax+1) - fmax
    t = np.arange(2*tmax+1) - tmax
    return np.outer(np.exp(logmodel(f, deltaf, exponent)), np.exp(logmodel(t, deltat, exponent)))

class Add_RFI_Plugin():
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self):
        params = self.ctx.params
        fit_sigma_freq = np.poly1d(params.coeff_freq)
        fit_sigma_time = np.poly1d(params.coeff_time)

        max_rfi_count = (self.ctx.tod_vx.shape[1] * self.ctx.params.strategy_step_size) / 3600 * params.max_rfi_count
        
        rfi_count = np.floor(np.random.uniform(1, max_rfi_count)).astype(int) 
        
        for rfi_idx in range(rfi_count):
            amp = np.fabs(stats.lognorm.rvs(1, loc=params.amp_loc, scale=params.amp_scale, size=1))
            sigma_freq = fit_sigma_freq(amp)
            sigma_time = fit_sigma_time(amp)
#             print("amp, sigma_freq, sigma_time", amp, sigma_freq, sigma_time)
             
            grid_x = np.arange(-params.sigma_range * sigma_time, params.sigma_range * sigma_time)
            grid_y = np.arange(-params.sigma_range * sigma_freq, params.sigma_range * sigma_freq)
            X,Y = np.meshgrid(grid_x,grid_y)
             
#             time_offset = np.random.normal(0, 1)
            time_offset = np.random.uniform(-params.sigma_range, params.sigma_range)
#             time_offset = sigma_range if np.fabs(time_offset) > sigma_range else time_offset
            Z = gaussian(amp, time_offset * sigma_time, 0, sigma_time, sigma_freq)(X,Y)
             
            pos_time = int(np.floor(np.random.uniform(0, self.ctx.tod_vx.shape[1] - 2 * params.sigma_range * sigma_time)))
            pos_freq = int(np.floor(np.random.uniform(0, self.ctx.tod_vx.shape[0] - 2 * params.sigma_range * sigma_freq)))
             
            if pos_time >= 0 and pos_freq >= 0:
                self.ctx.tod_vx[pos_freq: pos_freq + Z.shape[0], pos_time: pos_time + Z.shape[1]] += Z


        #constant
        for rfi_freq in params.rfi_freqs:
            amp = np.random.uniform(params.min_amp, params.max_amp)
#             print("amp", amp)
            scales1 = amp * np.exp(-np.arange(params.rfi_width-1, 0, -1) * 1.0)
            scales2 = amp * np.exp(-np.arange(0, params.rfi_width,  1) * 0.8)
            scales = np.append(scales1, scales2)
#             print("scales", scales)
            for i, rfi_pos in enumerate(np.arange(params.rfi_width-1, -params.rfi_width, -1)):
                scale = scales[i]
                rfi = np.random.normal(scale, scale, self.ctx.tod_vx.shape[1])
                self.ctx.tod_vx[rfi_freq - rfi_pos, : ] += rfi
        
        return self.ctx.tod_vx
    
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def testAddRfiPhaseswitch(target):       
    if target == 'default':
        nf = 300
        nt = 1800 # Default 600
    elif target == 'lofar':	
        nf = 64
        nt = 256
    tod = np.zeros((nf, nt))
    
    wn = np.arange(nf) + 1
    frac = .2 * np.ones(nf)
    Amax = 10 * wn
    
    params = Struct(white_noise_scale = wn, rfiamplitude = Amax, rfideltaf = rfideltaf,
                    rfideltat = rfideltat, rfifrac = frac, load_rfi_template = load_rfi_template,
                    rfiexponent = rfiexponent, rfienhance = rfienhance)
    
    strategy = [CoordSpec(t,0,0,0,0) for t in range(nt)]
    strategy_start = datetime(2016,1,1)
    ctx = Struct(params = params, tod_vx = tod.copy(), frequencies = wn, strategy_coords = strategy,
                    strategy_start = strategy_start)
    

    ctx.params.rfiday = rfiday#(3.0,23.0)
    ctx.params.rfidamping = rfidamping
    
    plugin = Plugin_Add_RFI_Phaseswitch(ctx)
    assert np.allclose(plugin.getTime(), np.arange(nt)/60./60.)
    rfi = plugin()
    return rfi

def testAddRfi(target):
    if target == 'default':
        tod = np.zeros((300, 1800)) # default: 300, 3600
        constantRfiFreqs = [25, 120]
        params = Struct(strategy_step_size=1,
                #bursts
                max_rfi_count = 20,
                coeff_freq = [0.179, 1.191],
                coeff_time = [0.144, -1.754, 63.035],
                sigma_range = 3.0,
                amp_scale = 3,
                amp_loc = 0,
                #constant
                rfi_freqs = constantRfiFreqs,
                min_amp = 1.,
                max_amp = 5.,
                rfi_width = 8,
                )
    elif target == 'lofar':
        tod = np.zeros((64, 256))
        constantRfiFreqs = [5, 34]
        
        params = Struct(strategy_step_size=1,
                        #bursts
                        max_rfi_count = 20,
                        coeff_freq = [0.179, 1.191],
                        coeff_time = [0.144, -1.754, 40],
                        sigma_range = 2.0,
                        amp_scale = 4,
                        amp_loc = 0,
                        #constant
                        rfi_freqs = constantRfiFreqs,
                        min_amp = 1.,
                        max_amp = 5.,
                        rfi_width = 8,
                        )
    
    ctx = Struct(params = params,
                    tod_vx = tod,
                    tod_vy = tod.copy())
    
    plugin = Add_RFI_Plugin(ctx)
    rfi = plugin()
    return rfi

def Simulate(plotLocation, target = 'lofar'):
    rfiGain = 10
    rfi = testAddRfi(target)
    rfi = np.clip(rfi,0,None)
    rfi = (rfi - np.min(rfi))/(np.max(rfi)*rfiGain - np.min(rfi))

    rfiPhaseSwitch = testAddRfiPhaseswitch(target)
    rfiPhaseSwitch = np.clip(rfiPhaseSwitch,0,None)
    rfiPhaseSwitch = (rfiPhaseSwitch - np.min(rfiPhaseSwitch))/(np.max(rfiPhaseSwitch)*rfiGain - np.min(rfiPhaseSwitch))

    rfi += rfiPhaseSwitch
    rfi = np.clip(rfi,0,1)

    plt.figure(figsize=(15,30))
    plt.imshow(rfi)
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plotLocation,"HIDE_{}_rfi.png".format(target)), dpi=300, bbox_inches='tight')
    plt.close()