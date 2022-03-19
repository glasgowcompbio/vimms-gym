import pandas as pd

from vimms.Exclusion import TopNExclusion


class Feature():
    def __init__(
        self, mz, rt, original_intensity, scaled_intensity, 
        fragmented, excluded):
        self.mz = mz
        self.rt = rt
        self.scaled_intensity = scaled_intensity
        self.original_intensity = original_intensity
        self.fragmented = fragmented
        self.excluded = excluded

    def __repr__(self):
        return f'mz={self.mz} rt={self.rt} intensity={self.original_intensity} ({self.scaled_intensity})'


class CleanerTopNExclusion(TopNExclusion):

    def __init__(self, mz_tol, rt_tol, initial_exclusion_list=None):
        super().__init__(initial_exclusion_list=initial_exclusion_list)
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol

    def update(self, mz, rt):
        if mz > 0 and rt > 0:
            x = self._get_exclusion_item(mz, rt, self.mz_tol, self.rt_tol)
            self.exclusion_list.add_box(x)


def obs_to_dfs(obs):
    scan_obs = {}
    count_obs = {}
    for key in obs:
        if key == 'valid_actions':
            continue
        val = obs[key]
        if len(val) == 1:
            count_obs[key] = val
        else:
            scan_obs[key] = val

    scan_df = pd.DataFrame(scan_obs)
    count_df = pd.DataFrame(count_obs)
    count_df = count_df.transpose()
    count_df.rename(columns={0: 'counts'}, inplace=True)
    return scan_df, count_df