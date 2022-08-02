import os
from os.path import exists

import numpy as np
from loguru import logger
from vimms.ChemicalSamplers import MZMLFormulaSampler, MZMLRTandIntensitySampler, \
    GaussianChromatogramSampler
from vimms.Common import POSITIVE, load_obj, save_obj


def preset_qcb_small():
    max_peaks = 100
    mzml_filename = os.path.abspath(os.path.join('..', 'notebooks', 'fullscan_QCB.mzML'))
    samplers_pickle = 'samplers_QCB_small.p'
    n_chemicals = (20, 50)
    mz_range = (100, 110)
    rt_range = (400, 500)
    intensity_range = (1E4, 1E20)
    params = generate_params(mzml_filename, samplers_pickle, n_chemicals,
                             mz_range, rt_range, intensity_range)
    return params, max_peaks


def preset_qcb_medium():
    max_peaks = 200
    mzml_filename = os.path.abspath(os.path.join('..', 'notebooks', 'fullscan_QCB.mzML'))
    samplers_pickle = 'samplers_QCB_medium.p'
    n_chemicals = (20, 50)
    mz_range = (100, 110)
    rt_range = (400, 500)
    intensity_range = (1E4, 1E20)
    params = generate_params(mzml_filename, samplers_pickle, n_chemicals,
                             mz_range, rt_range, intensity_range)
    return params, max_peaks


def preset_qcb_large():
    max_peaks = 200
    mzml_filename = os.path.abspath(os.path.join('..', 'notebooks', 'fullscan_QCB.mzML'))
    samplers_pickle = 'samplers_QCB_large.p'
    n_chemicals = (2000, 5000)
    mz_range = (70, 1000)
    rt_range = (0, 1440)
    intensity_range = (1E4, 1E20)
    params = generate_params(mzml_filename, samplers_pickle, n_chemicals,
                             mz_range, rt_range, intensity_range)
    return params, max_peaks


def generate_params(mzml_filename, samplers_pickle, n_chemicals, mz_range, rt_range,
                    intensity_range):
    min_mz = mz_range[0]
    max_mz = mz_range[1]
    min_rt = rt_range[0]
    max_rt = rt_range[1]
    min_log_intensity = np.log(intensity_range[0])
    max_log_intensity = np.log(intensity_range[1])
    isolation_window = 0.7
    rt_tol = 120
    mz_tol = 10
    ionisation_mode = POSITIVE
    enable_spike_noise = True
    noise_density = 0.1
    noise_max_val = 1E3
    mz_sampler, ri_sampler, cr_sampler = get_samplers(mzml_filename, samplers_pickle, min_mz,
                                                      max_mz, min_rt, max_rt, min_log_intensity,
                                                      max_log_intensity)
    params = {
        'chemical_creator': {
            'mz_range': mz_range,
            'rt_range': rt_range,
            'intensity_range': intensity_range,
            'n_chemicals': n_chemicals,
            'mz_sampler': mz_sampler,
            'ri_sampler': ri_sampler,
            'cr_sampler': cr_sampler,
        },
        'noise': {
            'enable_spike_noise': enable_spike_noise,
            'noise_density': noise_density,
            'noise_max_val': noise_max_val,
            'mz_range': mz_range
        },
        'env': {
            'ionisation_mode': ionisation_mode,
            'rt_range': rt_range,
            'isolation_window': isolation_window,
            'mz_tol': mz_tol,
            'rt_tol': rt_tol,
        }
    }
    return params


def get_samplers(mzml_filename, samplers_pickle, min_mz, max_mz, min_rt, max_rt, min_log_intensity,
                 max_log_intensity):
    if exists(samplers_pickle):
        logger.info('Loaded %s' % samplers_pickle)
        samplers = load_obj(samplers_pickle)
        mz_sampler = samplers['mz']
        ri_sampler = samplers['rt_intensity']
        cr_sampler = samplers['chromatogram']
    else:
        logger.info('Creating samplers from %s' % mzml_filename)
        mz_sampler = MZMLFormulaSampler(mzml_filename, min_mz=min_mz, max_mz=max_mz)
        ri_sampler = MZMLRTandIntensitySampler(mzml_filename, min_rt=min_rt, max_rt=max_rt,
                                               min_log_intensity=min_log_intensity,
                                               max_log_intensity=max_log_intensity)
        cr_sampler = GaussianChromatogramSampler()
        samplers = {
            'mz': mz_sampler,
            'rt_intensity': ri_sampler,
            'chromatogram': cr_sampler
        }
        save_obj(samplers, samplers_pickle)
    return mz_sampler, ri_sampler, cr_sampler
