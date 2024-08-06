import os
import warnings

from matplotlib import pyplot as plt
import numpy as np
import astropy.units as u
from pathlib import Path

def _listify(x):
    """Ensure a scalar/list is returned as a list.

    Taken from https://stackoverflow.com/a/1416677/1467820

    Copied from the pre-v1 hera_sim.rfi module.
    """
    try:
        basestring
    except NameError:
        basestring = (str, bytes)

    if isinstance(x, basestring):
        return [x]
    else:
        try:
            iter(x)
        except TypeError:
            return [x]
        else:
            return list(x)

class RfiStation:
    """Generate RFI based on a particular "station".

    Parameters
    ----------
    f0 : float
        Frequency that the station transmits (any units are fine).
    duty_cycle : float, optional
        With ``timescale``, controls how long the station is seen as "on". In
        particular, ``duty_cycle`` specifies which parts of the station's cycle are
        considered "on". Can be considered roughly a percentage of on time.
    strength : float, optional
        Mean magnitude of the transmission.
    std : float, optional
        Standard deviation of the random RFI magnitude.
    timescale : float, optional
        Controls the length of a transmision "cycle". Low points in the sin-wave cycle
        are considered "off" and high points are considered "on" (just how high is
        controlled by ``duty_cycle``). This is the wavelength (in seconds) of that
        cycle.
    rng: np.random.Generator, optional
        Random number generator.

    Notes
    -----
    This creates RFI with random magnitude in each time bin based on a normal
    distribution, with custom strength and variability. RFI is assumed to exist in one
    frequency channel, with some spillage into an adjacent channel, proportional to the
    distance to that channel from the station's frequency. It is not assumed to be
    always on, but turns on for some amount of time at regular intervals.
    """

    def __init__(self,f0: float, duty_cycle: float = 1.0, strength: float = 100.0, std: float = 10.0, timescale: float = 100.0):
        self.f0 = f0
        self.duty_cycle = duty_cycle
        self.strength = strength
        self.std = std
        self.timescale = timescale
        #self.rng = rng or np.random.default_rng()

    def __call__(self, lsts, freqs):
        """Compute the RFI for this station.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in units of ``f0``.


        Returns
        -------
        array-like
            2D array of RFI magnitudes as a function of LST and frequency.
        """
        # initialize an array for storing the rfi
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        # get the mean channel width
        channel_width = np.mean(np.diff(freqs))

        # find out if the station is in the observing band
        try:
            ch1 = np.argwhere(np.abs(freqs - self.f0) < channel_width)[0, 0]
        except IndexError:
            # station is not observed
            return rfi

        # find out whether to use the channel above or below... why?
        # I would think that the only time we care about neighboring
        # channels is when the station bandwidth causes the signal to
        # spill over into neighboring channels
        ch2 = ch1 + 1 if self.f0 > freqs[ch1] else ch1 - 1

        # generate some random phases
        phs1, phs2 = np.random.uniform(0, 2 * np.pi, size=2)

        # find out when the station is broadcasting
        is_on = 0.999 * np.cos(lsts * u.sday.to("s") / self.timescale + phs1)
        is_on = is_on > (1 - 2 * self.duty_cycle)

        # generate a signal and filter it according to when it's on
        signal = np.random.normal(self.strength, self.std, lsts.size)
        signal = np.where(is_on, signal, 0) * np.exp(1j * phs2)

        # now add the signal to the rfi array
        for ch in (ch1, ch2):
            # note: this assumes that the signal is completely contained
            # within the two channels ch1 and ch2; for very fine freq
            # resolution, this will usually not be the case
            df = np.abs(freqs[ch] - self.f0)
            taper = (1 - df / channel_width).clip(0, 1)
            rfi[:, ch] += signal * taper

        return rfi

class Stations():
    """A collection of RFI stations.

    Generates RFI from all given stations.

    Parameters
    ----------
    stations : list of :class:`RfiStation`
        The list of stations that produce RFI.
    rng: np.random.Generator, optional
        Random number generator.
    """

    _alias = ("rfi_stations",)
    is_randomized = True
    return_type = "per_baseline"

    def __init__(self, stations=None):
        self.stations = stations
        pass

        #super().__init__(stations=stations)

    def __call__(self, lsts, freqs):#, **kwargs):
        """Generate the RFI from all stations.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in units of ``f0`` for each station.

        Returns
        -------
        array-like of float
            2D array of RFI magnitudes as a function of LST and frequency.

        Raises
        ------
        TypeError
            If input stations are not of the correct type.
        """
        # kind of silly to use **kwargs with just one optional parameter...
        #self._check_kwargs(**kwargs)

        # but this is where the magic comes in (thanks to defaults)
        #(stations, rng) = self._extract_kwarg_values(**kwargs)

        # initialize an array to store the rfi in
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        if self.stations is None:
            warnings.warn(
                "You did not specify any stations to simulate.",
                stacklevel=2,
            )
            return rfi
        elif isinstance(self.stations, (str, Path)):
            # assume that it's a path to a npy file
            self.stations = np.load(self.stations)

        for station in self.stations:
            if not isinstance(station, RfiStation):
                if len(station) != 5:
                    raise ValueError(
                        "Stations are specified by 5-tuples. Please "
                        "check the format of your stations."
                    )

                # make an RfiStation if it isn't one
                station = RfiStation(*station)

            # add the effect
            rfi += station(lsts, freqs)

        return rfi

class Impulse():
    """Generate RFI impulses (short time, broad frequency).

    Parameters
    ----------
    impulse_chance : float, optional
        The probability in any given LST that an impulse RFI will occur.
    impulse_strength : float, optional
        Strength of the impulse. This will not be randomized, though a phase
        offset as a function of frequency will be applied, and will be random
        for each impulse.
    rng: np.random.Generator, optional
        Random number generator.
    """

    _alias = ("rfi_impulse",)
    is_randomized = True
    return_type = "per_baseline"

    def __init__(self, impulse_chance=0.001, impulse_strength=20.0, fixedTime = None):
        self.chance = impulse_chance
        self.strength = impulse_strength
        self.fixedTime = fixedTime

    def __call__(self, lsts, freqs):#, **kwargs):
        """Generate the RFI.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in arbitrary units.

        Returns
        -------
        array-like of float
            2D array of RFI magnitudes as a function of LST and frequency.
        """
        rng = np.random.default_rng()

        # initialize the rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        # find times when an impulse occurs
        if self.fixedTime is None:
            impulses = np.where(rng.uniform(size=lsts.size) <= self.chance)[0]
        else:
            impulses = np.asarray(self.fixedTime)

        # only do something if there are impulses
        if impulses.size > 0:
            # randomly generate some delays for each impulse
            dlys = rng.uniform(-300, 300, impulses.size)  # ns

            # generate the signals
            signals = self.strength * np.asarray([np.exp(2j * np.pi * dly * freqs) for dly in dlys])

            rfi[impulses] += signals

        return rfi

class Scatter():
    """Generate random RFI scattered around the waterfall.

    Parameters
    ----------
    scatter_chance : float, optional
        Probability that any LST/freq bin will be occupied by RFI.
    scatter_strength : float, optional
        Mean strength of RFI in any bin (each bin will receive its own
        random strength).
    scatter_std : float, optional
        Standard deviation of the RFI strength.
    rng: np.random.Generator, optional
        Random number generator.
    """

    _alias = ("rfi_scatter",)
    is_randomized = True
    return_type = "per_baseline"

    def __init__(self, scatter_chance=0.0001, scatter_strength=10.0, scatter_std=10.0):
        self.chance = scatter_chance
        self.strength = scatter_strength
        self.std = scatter_std

    def __call__(self, lsts, freqs):#, **kwargs):
        """Generate the RFI.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in arbitrary units.

        Returns
        -------
        array-like of float
            2D array of RFI magnitudes as a function of LST and frequency.
        """
        # now unpack them
        rng = np.random.default_rng()

        # make an empty rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        # find out where to put the rfi
        rfis = np.where(rng.uniform(size=rfi.size) <= self.chance)[0]

        # simulate the rfi; one random amplitude, all random phases
        signal = rng.normal(self.strength, self.std) * np.exp(2j * np.pi * rng.uniform(size=rfis.size))

        # add the signal to the rfi
        rfi.flat[rfis] += signal

        return rfi

class DTV():
    """Generate RFI arising from digitial TV channels.

    Digitial TV is assumed to be reasonably broad-band and scattered in time.

    Parameters
    ----------
    dtv_band : tuple, optional
        Lower edges of each of the DTV bands.
    dtv_channel_width : float, optional
        Channel width in GHz.
    dtv_chance : float, optional
        Chance that any particular time will have DTV.
    dtv_strength : float, optional
        Mean strength of RFI.
    dtv_std : float, optional
        Standard deviation of RFI strength.
    rng: np.random.Generator, optional
        Random number generator.
    """

    _alias = ("rfi_dtv",)
    is_randomized = True
    return_type = "per_baseline"

    def __init__(self,dtv_band=(0.174, 0.214),dtv_channel_width=0.008,dtv_chance=0.0001,dtv_strength=10.0,dtv_std=10.0):
        self.dtv_band = dtv_band
        self.width=dtv_channel_width
        self.dtv_chance = dtv_chance
        self.dtv_strength = dtv_strength
        self.dtv_std = dtv_std

    def __call__(self, lsts, freqs):#, **kwargs):
        """Generate the RFI.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in GHz.

        Returns
        -------
        array-like of float
            2D array of RFI magnitudes as a function of LST and frequency.
        """
        rng = np.random.default_rng()

        # make an empty rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        # get the lower and upper frequencies of the DTV band
        freq_min, freq_max = self.dtv_band

        # get the lower frequencies of each subband
        bands = np.arange(freq_min, freq_max, self.width)

        # if the bands fit exactly into the observed freqs, then we
        # need to ignore the uppermost DTV band
        if freqs.max() <= bands.max():
            bands = bands[:-1]

        # listify the listifiable parameters
        self.dtv_chance, self.dtv_strength, self.dtv_std = self._listify_params(bands, self.dtv_chance, self.dtv_strength, self.dtv_std)

        # find out which DTV channels will actually be observed
        overlap = np.logical_and(bands >= freqs.min() - self.width, bands <= freqs.max())

        # modify the bands and the listified parameters
        bands = bands[overlap]
        self.dtv_chance = self.dtv_chance[overlap]
        self.dtv_strength = self.dtv_strength[overlap]
        self.dtv_std = self.dtv_std[overlap]

        # raise a warning if there are no remaining bands
        if len(bands) == 0:
            warnings.warn(
                "The DTV band does not overlap with any of the passed "
                "frequencies. Please ensure that you are passing the "
                "correct set of parameters.",
                stacklevel=2,
            )

        # define an iterator, just to keep things neat
        df = np.mean(np.diff(freqs))
        dtv_iterator = zip(bands, self.dtv_chance, self.dtv_strength, self.dtv_std)

        # TODO: update the documentation here to make it more clear what's happening.
        # loop over the DTV bands, generating rfi where appropriate
        for band, chance, strength, std in dtv_iterator:
            # Find the first channel affected.
            if any(np.isclose(band, freqs, atol=0.01 * df)):
                ch1 = np.argwhere(np.isclose(band, freqs, atol=0.01 * df)).flatten()[0]
            else:
                ch1 = np.argwhere(band <= freqs).flatten()[0]
            try:
                # Find the last channel affected.
                if any(np.isclose(band + self.width, freqs, atol=0.01 * df)):
                    ch2 = np.argwhere(
                        np.isclose(band + self.width, freqs, atol=0.01 * df)
                    ).flatten()[0]
                else:
                    ch2 = np.argwhere(band + self.width <= freqs).flatten()[0]
                if ch2 == freqs.size - 1:
                    raise IndexError
            except IndexError:
                # in case the upper edge of the DTV band is outside
                # the range of observed frequencies
                ch2 = freqs.size

            # pick out just the channels affected
            this_rfi = rfi[:, ch1:ch2]

            # find out which times are affected
            rfis = rng.uniform(size=lsts.size) <= chance

            # calculate the signal
            signal = np.atleast_2d(rng.normal(strength, std, size=rfis.sum())* np.exp(2j * np.pi * rng.uniform(size=rfis.sum()))).T

            # add the signal to the rfi array
            this_rfi[rfis] += signal
        return rfi

    def _listify_params(self, bands, *args):
        Nchan = len(bands)
        listified_params = []
        for arg in args:
            # ensure that the parameter is a list
            # if not isinstance(arg, (list, tuple)):
            #     arg =
            arg = _listify(arg)

            # update the length if it's a singleton
            if len(arg) == 1:
                arg *= Nchan

            # check that the length matches the number of DTV bands
            if len(arg) != Nchan:
                raise ValueError(
                    "At least one of the parameter values for "
                    "dtv_chance, dtv_strength, or dtv_std is not "
                    "formatted properly. These parameters must satisfy "
                    "*one* of the following conditions: \n"
                    "Only a single value is specified *OR* a list of "
                    "values with the same length as the number of DTV "
                    "bands specified. For reference, the DTV bands you "
                    "specified have the following characteristics: \n"
                    f"f_min : {bands[0]} \nf_max : {bands[-1]}\n N_bands : "
                    f"{Nchan}"
                )

            # everything should be in order now, so
            listified_params.append(np.asarray(arg))

        return listified_params

def Simulate(plotLocation, target = 'lofar'):
    if target == 'default':      
        pImpulse = 0.001 # 0.001
        pScatter = 0.0001 # 0.0001
        pDtv = 0.01 # 0.0001
        lsts = np.linspace(0, 1799, 1800)
        freqs = np.linspace(0, 299, 300)
        rfi_stations = Stations([RfiStation(50, duty_cycle=0.1,timescale=50),RfiStation(100)])
        rfi_impulse = Impulse(impulse_chance=pImpulse)
        rfi_scatter = Scatter(scatter_chance=pScatter)
        rfi_dtv = DTV(dtv_band=(150,200),dtv_channel_width=4,dtv_chance=pDtv)
    elif target == 'lofar':
        pImpulse = 0.001 # 0.001
        pScatter = 0.005 # 0.0001
        pDtv = 0.01 # 0.0001
        lsts = np.linspace(0, 255, 256)
        freqs = np.linspace(0, 63, 64)
        rfi_stations = Stations([RfiStation(10, duty_cycle=0.1,timescale=50),RfiStation(5)])
        rfi_impulse = Impulse(impulse_chance=pImpulse, fixedTime = [150,200])
        rfi_scatter = Scatter(scatter_chance=pScatter)
        rfi_dtv = DTV(dtv_band=(20,30),dtv_channel_width=2,dtv_chance=pDtv)
    
    allComponentsFound = False
    while allComponentsFound == False:
        stationsRfi = rfi_stations(lsts, freqs)
        impulseRfi = rfi_impulse(lsts, freqs)
        scatterRfi = rfi_scatter(lsts, freqs)
        dtvRfi = rfi_dtv(lsts, freqs)
        
        if target == 'lofar':
            scatterRfi[:,:35] = 0

        stationsMagnitude = np.abs(stationsRfi).T
        impulseMagnitude = np.abs(impulseRfi).T
        scatterMagnitude = np.abs(scatterRfi).T
        dtvMagnitude = np.abs(dtvRfi).T
        
        allComponentsFound = True
        if np.max(stationsMagnitude)==0:
            allComponentsFound = False
        if np.max(impulseMagnitude)==0:
            allComponentsFound = False
        if np.max(scatterMagnitude)==0:
            allComponentsFound = False
        if np.max(dtvMagnitude)==0:
            allComponentsFound = False

        stationsMagnitude = (stationsMagnitude - np.min(stationsMagnitude))/(np.max(stationsMagnitude) - np.min(stationsMagnitude))
        impulseMagnitude = (impulseMagnitude - np.min(impulseMagnitude))/(np.max(impulseMagnitude) - np.min(impulseMagnitude))
        scatterMagnitude = (scatterMagnitude - np.min(scatterMagnitude))/(np.max(scatterMagnitude) - np.min(scatterMagnitude))
        dtvMagnitude = (dtvMagnitude - np.min(dtvMagnitude))/(np.max(dtvMagnitude) - np.min(dtvMagnitude))

        stationsPhase = np.angle(stationsRfi).T
        impulsePhase = np.angle(impulseRfi).T
        scatterPhase = np.angle(scatterRfi).T
        dtvPhase = np.angle(dtvRfi).T

    rfi = stationsMagnitude+impulseMagnitude + scatterMagnitude + dtvMagnitude
    rfi = np.clip(rfi,0,1)
    rfiPhase = stationsPhase+impulsePhase + scatterPhase + dtvPhase
    rfiPhase  = (rfiPhase + np.pi)/(2*np.pi)

    plt.figure(figsize=(15,30))
    plt.imshow(rfi)
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.ylim(0,freqs[-1])
    plt.xlim(0, lsts[-1])
    plt.savefig(os.path.join(plotLocation,"heraSim_{}_rfi.png".format(target)), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15,30))
    plt.imshow(rfiPhase)
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.ylim(0,freqs[-1])
    plt.xlim(0, lsts[-1])
    plt.savefig(os.path.join(plotLocation,"heraSim_{}_rfi_phase.png".format(target)), dpi=300, bbox_inches='tight')
    plt.close()