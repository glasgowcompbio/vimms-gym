import numpy as np
import pandas as pd
from vimms.Exclusion import TopNExclusion


class Feature():
    def __init__(
            self, mz, rt, intensity, fragmented, excluded, roi):
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.fragmented = fragmented
        self.excluded = excluded
        self.roi = roi

    def __repr__(self):
        return f'mz={self.mz} rt={self.rt} ' \
               f'intensity={self.intensity} ' \
               f'fragmented={self.fragmented} ' \
               f'roi={self.roi}'


class CleanerTopNExclusion(TopNExclusion):

    def __init__(self, mz_tol, rt_tol, initial_exclusion_list=None):
        super().__init__(initial_exclusion_list=initial_exclusion_list)
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol

    def update(self, mz, rt):
        if mz > 0 and rt > 0:
            x = self._get_exclusion_item(mz, rt, self.mz_tol, self.rt_tol)
            self.exclusion_list.add_box(x)


def obs_to_dfs(obs, features):
    scan_obs = {}
    count_obs = {}
    for key in obs:
        if key == 'valid_actions':
            continue
        val = obs[key]
        try:
            if len(val) == 1:
                count_obs[key] = val
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                # skip 2D NumPy arrays
                pass
            else:
                scan_obs[key] = val
        except TypeError: # no length, so must be a scalar
            count_obs[key] = val

    scan_df = pd.DataFrame(scan_obs)

    # set the original log intensity values to scan_df too
    log_intensities = np.zeros(len(scan_df))
    for i in range(len(features)):
        f = features[i]
        log_intensities[i] = np.log(f.intensity)
    scan_df['log_intensities'] = log_intensities

    # create a dataframe to hold various counts
    count_df = pd.DataFrame(count_obs)
    count_df = count_df.transpose()
    count_df.rename(columns={0: 'counts'}, inplace=True)

    return scan_df, count_df
