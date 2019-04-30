from pathlib import Path

from astropy.table import Table
from numba import njit, prange
from numpy import atleast_2d, zeros, log, concatenate, pi, transpose, sum
from numpy.polynomial.legendre import legvander
from numpy.random import uniform
from pytransit.lpf.oclttvlpf import OCLTTVLPF
from pytransit.param.parameter import LParameter, UniformPrior as UP, NormalPrior as NP
from pytransit.utils.keplerlc import KeplerLC


@njit(parallel=True, cache=False, fastmath=True)
def lnlike_normal_v(o, m, e):
    m = atleast_2d(m)
    npv = m.shape[0]
    npt = o.size
    lnl = zeros(npv)
    for i in prange(npv):
        lnl[i] = -npt*log(e[i, 0]) - 0.5*log(2*pi) - 0.5*sum(((o-m[i, :])/e[i ,0])**2)
    return lnl


class TESSTTVLPF(OCLTTVLPF):
    def __init__(self, name: str, dfile: Path, zero_epoch: float, period: float, nsamples: int = 10,
                 trdur: float = 0.125, bldur: float = 0.3, nlegendre: int = 2, ctx = None, queue = None):

        tb = Table.read(dfile)
        self.bjdrefi = tb.meta['BJDREFI']
        zero_epoch = zero_epoch - self.bjdrefi

        df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
        self.lc = lc = KeplerLC(df.TIME.values, df.PDCSAP_FLUX.values, zeros(df.shape[0]),
                                zero_epoch, period, trdur, bldur)

        self.nlegendre = nlegendre
        super().__init__(name, zero_epoch, period, ['TESS'],
                         times=lc.time_per_transit, fluxes=lc.normalized_flux_per_transit,
                         pbids=lc.nt * [0], nsamples=nsamples, exptimes=[0.00139],
                         cl_ctx=ctx, cl_queue=queue)

        self.mtimes = [t - t.mean() for t in self.times]
        self.windows = window = concatenate(self.mtimes).ptp()
        self.mtimes = [t / window for t in self.mtimes]
        self.legs = [legvander(t, self.nlegendre) for t in self.mtimes]
        self.ofluxa = self.ofluxa.astype('d')
        self.lnlikelihood = self.lnlikelihood_nb

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        for sl in self.ps.blocks[1].slices:
            pvp[:, sl] = uniform(0.01 ** 2, 0.05 ** 2, size=(npop, 1))
        return pvp

    def _init_p_noise(self):
        """Noise parameter initialisation.
        """
        pns = [LParameter('log_err', 'log_error', '', UP(-8, -0), bounds=(-8, -0))]
        self.ps.add_lightcurve_block('log_err', 1, 1, pns)
        self._sl_err = self.ps.blocks[-1].slice
        self._start_err = self.ps.blocks[-1].start

    def _init_p_baseline(self):
        """Baseline parameter initialisation.
        """
        bls = []
        for i, tn in enumerate(self.tnumber):
            bls.append(LParameter(f'bli_{tn}', f'bl_intercept_{tn}', '', NP(1.0, 0.01), bounds=(0.98, 1.02)))
            for ipoly in range(1, self.nlegendre + 1):
                bls.append(
                    LParameter(f'bls_{tn}_{ipoly}', f'bl_c_{tn}_{ipoly}', '', NP(0.0, 0.001), bounds=(-0.1, 0.1)))
        self.ps.add_lightcurve_block('baseline', self.nlegendre + 1, self.nlc, bls)
        self._sl_bl = self.ps.blocks[-1].slice
        self._start_bl = self.ps.blocks[-1].start

    def baseline(self, pvp):
        """Multiplicative baseline"""
        pvp = atleast_2d(pvp)
        fbl = zeros((pvp.shape[0], self.timea.size))
        for ipv, pv in enumerate(pvp):
            bl = pv[self._sl_bl]
            for itr, sl in enumerate(self.lcslices):
                fbl[ipv, sl] = bl[itr * (self.nlegendre + 1):(itr + 1) * (self.nlegendre + 1)] @ self.legs[itr].T
        return fbl

    def flux_model(self, pvp):
        tmodel = transpose(self.transit_model(pvp, copy=True)).copy()
        return tmodel * self.baseline(pvp)

    def lnlikelihood_nb(self, pvp):
        fmodel = self.flux_model(pvp).astype('d')
        err = 10 ** atleast_2d(pvp)[:, self._sl_err]
        return lnlike_normal_v(self.ofluxa, fmodel, err)
