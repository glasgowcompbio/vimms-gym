"""
Created on 16/06/2022 20:42
@author: Liu Ziyan
@E-mail: 2650906L@student.gla.ac.uk
"""
import time
import sys

sys.path.append('..')

from viewer_helper import run_simulation, load_model_and_params, get_parameters, METHOD_DQN_COV, METHOD_DQN_INT

from vimms_gym.common import METHOD_PPO, METHOD_TOPN, METHOD_RANDOM, METHOD_FULLSCAN, render_scan, \
    METHOD_DQN
from vimms_gym.common import MAX_OBSERVED_LOG_INTENSITY, MAX_ROI_LENGTH_SECONDS
from vimms_gym.chemicals import generate_chemicals

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


def main_window():
    st.set_page_config(layout="wide")
    st.title('VIMMS-Gym Viewer')
    st.header("Interactive visualisation of VIMMS-Gym model")
    environment = sidebar_environment()  # sidebar
    if st.sidebar.button('Generate chemicals'):  # button
        # run
        chems, max_peaks, params = Generate_chems(environment)
        # store
        st.session_state['environment'] = environment
        st.session_state['chems'] = chems
        st.session_state['max_peaks'] = max_peaks
        st.session_state['params'] = params
        st.session_state['sorted_chemicals'] = None
        st.session_state['sorted_chemicals_df'] = None
        st.session_state['results'] = {}
        st.session_state['store'] = {}
        st.session_state['flag'] = False
        st.session_state['link_data'] = []

    policy = sidebar_policy()  # sidebar
    # set trajectory params
    budget = None
    t_length = None
    statesAfter = None
    intervalSize = None
    if policy != METHOD_TOPN:
        budget, t_length, statesAfter, intervalSize = sidebar_trajectory_params()

    # button
    if st.sidebar.button('Run episode') and 'chems' in st.session_state and policy not in st.session_state['results']:
        info_container = st.empty()
        with info_container.container():
            # run training
            episode, env, T, random = train(environment, policy, st.session_state['params'], st.session_state['chems'],
                                            st.session_state['max_peaks'], budget, t_length, statesAfter, intervalSize)
            # store
            st.session_state['results'][policy] = [episode, env, T, random]
            st.session_state['store'][policy] = {}
        info_container.empty()
    elif 'chems' not in st.session_state:
        st.warning('Please generate chemicals first!!!')

    st.sidebar.write('')
    st.sidebar.write('')
    visual_window()


def sidebar_environment():
    st.sidebar.header("Simulation Parameters")
    st.sidebar.subheader('Generate chemicals')
    st.sidebar.write('Select a preset environment or upload your own chemicals')
    environment = None
    # env selection
    environment = st.sidebar.radio(
        "Select a preset environment",
        ('QCB_chems_small', 'QCB_chems_medium', 'QCB_resimulated_medium'))
    if environment == "QCB_chems_small":
        st.sidebar.caption('Generate 20 - 50 chemical objects with m/z, RT and intensities that resembles the QC '
                           'Beer sample. '
                           'Chromatographic peak shapes are assumed to be Gaussian.')
    elif environment == "QCB_chems_medium":
        st.sidebar.caption(
            'Generate 200 - 500 chemical objects with m/z, RT and intensities that resembles the QC Beer sample.'
            'Chromatographic peak shapes are assumed to be Gaussian.')
    elif environment == "QCB_resimulated_medium":
        st.sidebar.caption(
            'Generate 200 - 500 chemical objects with m/z, RT and intensities that resembles the QC Beer sample.'
            'Chromatographic peak shapes are extracted from experimental data by detecting regions-of-interest '
            '(ROIs) in the fullscan mzML file.')

    return environment


def sidebar_policy():
    st.sidebar.markdown("---")
    # policy selection
    st.sidebar.subheader("Fragmentation policy")
    policy = st.sidebar.radio(
        "Select a policy",
        (METHOD_TOPN, METHOD_DQN_COV, METHOD_DQN_INT))
    if policy == METHOD_TOPN:
        st.sidebar.caption(
            'This strategy is a typical ion fragmentation strategy of data-dependent acquisition in LC-MS/MS. '
            'It will preferentially extract the highest intenstity precursor ions for fragmentation.')
    elif policy == METHOD_DQN_COV:
        st.sidebar.caption('DQN is a reinforcement learning algorithm where a deep learning model is built '
                           'to find the actions an agent can take at each state. This policy focus on the main goal '
                           'that fragmenting as many ions as possible.')
    elif policy == METHOD_DQN_INT:
        st.sidebar.caption('DQN is a reinforcement learning algorithm where a deep learning model is built '
                           'to find the actions an agent can take at each state. This policy focus on the main goal '
                           'that fragmenting ions at a intensity as close as close as possible to the maximum '
                           'intensity of ion chromatography.')

    # if policy == METHOD_PPO:
    # model_file = st.sidebar.file_uploader('Upload pre-trained PPO model (StableBaselines3)')
    st.sidebar.markdown("---")
    return policy


def sidebar_trajectory_params():
    st.sidebar.subheader('Trajectory parameters')
    st.sidebar.caption('These parameters determine the total number of extracted trajectories, their length, '
                       'the length before and after the trajectory is extracted around a particular single timestep, '
                       'and how many timesteps are spaced between trajectories.')
    # input necessary
    budget = st.sidebar.number_input('Input number of trajectory', value=10)
    t_length = st.sidebar.number_input('Input the length of each trajectory', value=20)
    statesAfter = st.sidebar.number_input('Input the length of states after', value=10)
    intervalSize = st.sidebar.number_input('Input the interval size between trajectories', value=50)
    st.sidebar.markdown("---")
    return budget, t_length, statesAfter, intervalSize


def Generate_chems(environment):
    params, max_peaks = get_parameters(environment)
    if params is not None:
        # generate chemicals following the selected preset
        chemical_creator_params = params['chemical_creator']
        chems = generate_chemicals(chemical_creator_params)
        info_container = st.empty()
        with info_container.container():
            st.metric(label='', value="{0} chemicals have been generated".format(str(len(chems))))
            time.sleep(1.5)
        info_container.empty()
        return chems, max_peaks, params
    else:
        st.error("parameters error!")


def train(environment, policy, params, chems, max_peaks, budget, t_length, statesAfter, intervalSize):
    # trainning function
    if params is not None:
        # run simulation to generate an episode
        N, min_ms1_intensity, model, params = load_model_and_params(environment, policy, params)
        episode, env, T, random = run_simulation(N, chems, max_peaks, policy, min_ms1_intensity, model,
                                                 params, budget, t_length, statesAfter, intervalSize)
        return episode, env, T, random
    else:
        st.error("parameters error!")


def visual_window():  # interpretation part
    if 'results' in st.session_state and st.session_state['results'] != {}:
        st.header('Explore episode')
        # set up the placeholder
        explore_window = st.empty()
        plots_window = st.empty()
        # get data which been stored
        environment = st.session_state['environment']
        result = explore_window.selectbox('Select results', st.session_state['results'].keys())
        episode = st.session_state['results'][result][0]
        env = st.session_state['results'][result][1]
        max_peaks = st.session_state['max_peaks']
        # methods selection
        vm = plots_window.selectbox('Select visualization methods',
                                    ('view features', 'view chemicals', 'view trajectory', 'random trajectory'))
        st.markdown("---")
        if 'ms2_frags' not in st.session_state['store'][result]:
            # get fragment info
            st.session_state['store'][result]['ms2_frags'] = [
                e for e in env.vimms_env.mass_spec.fragmentation_events if e.ms_level == 2]
        # extract data and store
        if 'feature' not in st.session_state['store'][result]:
            index = []
            ac = []
            re = []
            intensities = []
            excluded = []
            roi_length = []
            roi_elapsed_time_since_last_frag = []
            roi_intensity_at_last_frag = []
            roi_min_intensity_since_last_frag = []
            roi_max_intensity_since_last_frag = []
            scan_id = []
            # extract feature data
            for step in range(episode.num_steps):
                if episode.actions[step] != max_peaks and step < episode.num_steps - 1:
                    intensities.append(
                        round(episode.observations[step + 1]['intensities'][
                                  episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY, 2))
                    excluded.append(
                        round(episode.observations[step + 1]['excluded'][episode.actions[step]] * env.rt_tol, 2))
                    roi_length.append(
                        round(episode.observations[step + 1]['roi_length'][
                                  episode.actions[step]] * MAX_ROI_LENGTH_SECONDS, 2))
                    roi_elapsed_time_since_last_frag.append(
                        round(episode.observations[step + 1]['roi_elapsed_time_since_last_frag'][
                                  episode.actions[step]] * MAX_ROI_LENGTH_SECONDS, 2))
                    roi_intensity_at_last_frag.append(
                        round(episode.observations[step + 1]['roi_intensity_at_last_frag'][
                                  episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY, 2))
                    roi_min_intensity_since_last_frag.append(
                        round(episode.observations[step + 1]['roi_min_intensity_since_last_frag'][
                                  episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY, 2))
                    roi_max_intensity_since_last_frag.append(
                        round(episode.observations[step + 1]['roi_max_intensity_since_last_frag'][
                                  episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY, 2))
                    ac.append(episode.actions[step])
                    re.append(round(episode.rewards[step], 2))
                    index.append(step)
                    scan_id.append(episode.get_step_data(step + 1)['info']['current_scan_id'])
            # make dataframe
            df = pd.DataFrame(
                {'scan_id': scan_id, 'timestep': index, 'action': ac, 'reward': re, 'intensities': intensities,
                 'excluded': excluded,
                 'roi_length': roi_length, 'roi_elapsed_time_since_last_frag': roi_elapsed_time_since_last_frag,
                 'roi_intensity_at_last_frag': roi_intensity_at_last_frag,
                 'roi_min_intensity_since_last_frag': roi_min_intensity_since_last_frag,
                 'roi_max_intensity_since_last_frag': roi_max_intensity_since_last_frag})
            st.session_state['store'][result]['feature'] = df
        if st.session_state['sorted_chemicals'] is None or st.session_state['sorted_chemicals_df'] is None:
            # sort chemicals by max intensity
            sorted_chemicals = st.session_state['chems']
            for i in range(len(sorted_chemicals)):
                for j in range(len(sorted_chemicals) - 1 - i):
                    if sorted_chemicals[j].max_intensity < sorted_chemicals[j + 1].max_intensity:
                        sorted_chemicals[j], sorted_chemicals[j + 1] = sorted_chemicals[j + 1], sorted_chemicals[j]
            # make df with related info
            mz = [round(chem.mass, 4) for chem in sorted_chemicals]
            rt = [round(chem.rt, 2) for chem in sorted_chemicals]
            max_intensity = [round(chem.max_intensity, 2) for chem in sorted_chemicals]
            idx = [sorted_chemicals.index(chem) + 1 for chem in sorted_chemicals]
            chemical_df = pd.DataFrame(
                {'index': idx, 'mz': mz, 'rt': rt, 'max_intensity': max_intensity})
            # store
            st.session_state['sorted_chemicals'] = sorted_chemicals
            st.session_state['sorted_chemicals_df'] = chemical_df

        if vm == 'view features':
            st.header('View Features')
            st.caption('In VIMMS-Gym, the state consists of the feature vector of the precursor ion. '
                       'Below are the different features. '
                       'Buttons and sliders can be used to select features in different timesteps for observation.')
            # Radio to control feature selection
            feature = st.radio('select a feature',
                               ('intensities', 'excluded', 'roi_length', 'roi_elapsed_time_since_last_frag',
                                'roi_intensity_at_last_frag', 'roi_min_intensity_since_last_frag',
                                'roi_max_intensity_since_last_frag'), horizontal=True)
            # sliders to control range selection
            timestep_range = st.slider('select timestep range',
                                       0, episode.num_steps - 1, (0, episode.num_steps - 1))
            feature_range = st.slider('select feature range',
                                      round(min(st.session_state['store'][result]['feature'][feature]), 2),
                                      round(max(st.session_state['store'][result]['feature'][feature]), 2),
                                      (round(min(st.session_state['store'][result]['feature'][feature]), 2),
                                       round(max(st.session_state['store'][result]['feature'][feature]), 2)))
            # draw plots
            view_feature(result, feature, timestep_range, feature_range, link=False, link_data=None)

        elif vm == 'view chemicals':
            st.header('View Chemicals')
            # do function
            view_chemical(result)
            # control interactive function
            featureplot = st.empty()
            if not st.session_state['flag']:
                featureplot.empty()
            if st.session_state['flag']:
                with featureplot.container():
                    st.subheader('View timestep')
                    st.caption(
                        'The next two figures show the timestep and the surrounding timestep at which the selected '
                        'MS2 event occurs.')
                    view_feature(result, link=True, link_data=st.session_state['link_data'])
                st.session_state['flag'] = False

        elif vm == 'view trajectory':
            if result == 'topN':
                st.warning('topN model dose not have this function!!!')
            else:
                st.header('View Trajectories')
                st.caption('In a simulated episode, a trajectory represents a sequence of a particular timestep and a '
                           'number of timesteps around it. This sequence contains pairs of states and actions in '
                           'these timesteps. The trajectories provided here are extracted with reference to the '
                           'HIGHLIGHT algorithm. In a episode, some timesteps contain state and action pairs with '
                           'higher Q difference are considered as significant timesteps. These timesteps will be used '
                           'as the beginning of the extraction trajectory to extract the timesteps around it as a '
                           'sequence.')
                # extract trajectory info
                view_trajectory(result, max_peaks, env, random=False)
        elif vm == 'random trajectory':
            if environment == 'QCB_resimulated_medium':
                if result == 'topN':
                    st.warning('topN model dose not have this function!!!')
                else:
                    view_trajectory(result, max_peaks, env, random=True)
            else:
                st.warning(environment + ' does not has this function!!!')


def view_feature(result, feature=None, timestep_range=None, feature_range=None, link=False, link_data=None):
    if link is False:  # finish the fuction of feature part
        ms2_frags = st.session_state['store'][result]['ms2_frags']
        # df is reconstructed according to the selected feature to realize slider-based control of the image
        # reset the axis range
        plot_df = st.session_state['store'][result]['feature'][
            st.session_state['store'][result]['feature']['timestep'] > timestep_range[0]
            ]
        plot_df = plot_df[plot_df['timestep'] < timestep_range[1]]
        # filter by feature
        plot_df = plot_df[plot_df[feature] >= feature_range[0]]
        plot_df = plot_df[plot_df[feature] <= feature_range[1]]
        # draw plots
        st.caption("In VIMMS-Gym, the chemicals are ranked from 0 to max peak in descending order of intensity "
                   "in MS1 scan. Action is pick certain chemical to fragment. Reward is calculated according to "
                   "the intensity. The points on the graph are clickable. An information table and chromatogram of "
                   "the ions being fragmented associated with the timestep of the click are generated below. There "
                   "will be green lines on the chromatogram indicating when the fragmentation occurred. A timestep "
                   "may be associated with one or more fragmentation events.")
        col3, col4 = st.columns(2)
        with col3:
            fig_ac = px.scatter(plot_df, x='timestep', y='action',
                                title='Action vs timestep')
            # interactive plots
            selected_points_ac = plotly_events(fig_ac, click_event=True)
            if selected_points_ac:
                timestep = selected_points_ac[0]['x']
                action = selected_points_ac[0]['y']
                st.write('you select: timestep: ' + str(timestep) + ' ' + 'action: ' + str(action))
                click_df = plot_df[plot_df['timestep'] == timestep]['scan_id']
                scanid = click_df[click_df.index[0]]
                e = [event for event in ms2_frags if event.scan_id == scanid]
                click_dots(e)
        with col4:
            fig_re = px.scatter(plot_df, x='timestep', y='reward',
                                title='Reward vs timestep')
            # interactive plots
            selected_points_re = plotly_events(fig_re, click_event=True)
            if selected_points_re:
                timestep = selected_points_re[0]['x']
                reward = selected_points_re[0]['y']
                st.write('you select: timestep: ' + str(timestep) + ' ' + 'reward: ' + str(reward))
                click_df = plot_df[plot_df['timestep'] == timestep]['scan_id']
                scanid = click_df[click_df.index[0]]
                e = [event for event in ms2_frags if event.scan_id == scanid]
                click_dots(e)

    elif link is True:  # finish function related to chemical part
        plot_df = st.session_state['store'][result]['feature'].loc[
                  link_data.index[0] - 5:link_data.index[0] + 6, :]
        new_feature_vector = []
        for value in plot_df['timestep']:
            for link_value in link_data['timestep']:
                if value == link_value:
                    new_feature_vector.append('Ms2 event')
                else:
                    new_feature_vector.append('timestep around')
        plot_df.insert(loc=11, column='type', value=new_feature_vector)
        col3, col4 = st.columns(2)
        with col3:
            fig = px.scatter(plot_df, x='timestep', y='action', color='type')
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.scatter(plot_df, x='timestep', y='reward', color='type')
            st.plotly_chart(fig, use_container_width=True)


def click_dots(scan_events):
    st.write('related chemicals')
    # if too many chemicals are contained in a timestep just hide them. Or webpage will be broken
    if len(scan_events) > 5:
        st.write(str(len(scan_events)) + ' frag events at this timestep, 5 are shown')
    elif len(scan_events) <= 1:
        st.write(str(len(scan_events)) + ' frag event at this timestep')
    else:
        st.write(str(len(scan_events)) + ' frag events at this timestep')
    i = 0
    for e in scan_events:
        i += 1
        click_event_df = pd.DataFrame(
            {'mz': [e.chem.mass], 'rt': [e.chem.rt], 'max_intensity': [e.chem.max_intensity],
             'query_rt': [e.query_rt], 'scan_id': [e.scan_id]})
        st.write(click_event_df)
        chrom = e.chem.chromatogram
        X = []
        Y = []
        rt = chrom.min_rt
        while rt <= chrom.max_rt:
            X.append(rt + e.chem.rt)
            Y.append(chrom.get_relative_intensity(rt))
            rt += (chrom.max_rt - chrom.min_rt) / 19
        click_fig = px.line(x=X, y=Y, labels={'x': 'RT', 'y': 'intensity'})
        click_fig.add_vline(x=e.query_rt, line_width=1, line_dash="dash", line_color="green")
        st.plotly_chart(click_fig, use_container_width=True)
        if i == 5:
            break


def view_chemical(result):
    st.caption('All the generated chemicals are shown in the following table, select one of them to observe the '
               'fragmented information such as chromatogram and MS2 events.')
    st.caption('Coverage proportion measures how many chemicals a method is able to ‘hit’ with frag events.'
               'Intensity proportion measures whether method can ‘hit’ with frag events at Max peak.')
    ms2_frags = st.session_state['store'][result]['ms2_frags']
    cp_col, ip_col = st.columns(2)
    coverage_p, intensity_p = get_proportion(ms2_frags, st.session_state['sorted_chemicals'])
    cp_col.metric('Coverage proportion: ', value=coverage_p)
    ip_col.metric('Intensity proportion: ', value=intensity_p)

    # make table
    gd = GridOptionsBuilder.from_dataframe(st.session_state['sorted_chemicals_df'])
    gd.configure_side_bar()
    gd.configure_selection(selection_mode='single', use_checkbox=True)
    grid_table = AgGrid(st.session_state['sorted_chemicals_df'], enable_enterprise_modules=True, height=250,
                        gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED,
                        fit_columns_on_grid_load=True)
    if grid_table["selected_rows"]:
        selection = grid_table["selected_rows"][0]['index']
        # get chemicals
        chemical = st.session_state['sorted_chemicals'][selection - 1]
        # get related ms2 events
        frags = [event for event in ms2_frags if chemical == event.chem]
        # extract related RT and intensities
        chrom = chemical.chromatogram
        X = []
        Y = []
        rt = chrom.min_rt
        while rt <= chrom.max_rt:
            X.append(rt + chemical.rt)
            Y.append(chrom.get_relative_intensity(rt))
            rt += (chrom.max_rt - chrom.min_rt) / 29
        # draw plots
        if frags:
            st.subheader('View chromatogram of chemicals')
            st.caption('The blue line shown in the figure represents the chromatographic peak of the selected '
                       'chemical and the green dotted lines represent all MS2 events (fragmentation events) occurring '
                       'for the chemical.')
            fig = px.line(x=X, y=Y, labels={'x': 'RT', 'y': 'intensity'})
            for frag in frags:
                fig.add_vline(x=frag.query_rt, line_width=1, line_dash="dash", line_color="green")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader('Ms2 events info')
            st.caption(
                'This table shows the related fragmentation info of this chemical. Each row from top to bottom of the '
                'data in the table represents the MS2 event from left to right in the figure above.')
            # extract ms2 info and store in a df
            frag_df = []
            for frag in frags:
                store = st.session_state['store'][result]['feature'][
                    st.session_state['store'][result]['feature']['scan_id'] == frag.scan_id]
                store.insert(loc=0, column='query_rt', value=round(frag.query_rt, 4))
                frag_df.append(store)
            frag_df = pd.concat(frag_df)
            ms2_gd = GridOptionsBuilder.from_dataframe(frag_df)
            ms2_gd.configure_side_bar()
            ms2_gd.configure_selection(selection_mode='single', use_checkbox=True)
            ms2_grid_table = AgGrid(frag_df, enable_enterprise_modules=True, height=300, gridOptions=ms2_gd.build(),
                                    update_mode=GridUpdateMode.SELECTION_CHANGED)
            if ms2_grid_table["selected_rows"]:  # store and pass the data to view_feature(link=True)
                ms2_selection = ms2_grid_table["selected_rows"][0]['scan_id']
                link_data = st.session_state['store'][result]['feature'][
                    st.session_state['store'][result]['feature']['scan_id'] == ms2_selection]
                st.session_state['flag'] = True
                st.session_state['link_data'] = link_data

        elif not frags:
            st.warning('This chemical is not fragmented!!!')


def view_trajectory(result, max_peaks, env, random=False):
    # extract trajectory objects and get info to show in a df
    feature_name = ['excluded', 'roi_length', 'roi_elapsed_time_since_last_frag',
                    'roi_intensity_at_last_frag', 'roi_min_intensity_since_last_frag',
                    'roi_max_intensity_since_last_frag', 'intensities']
    if not random:
        T = st.session_state['results'][result][2]
    elif random:
        T = st.session_state['results'][result][3]
    t_table = pd.DataFrame({'trajectory': ['trajectory ' + str(i + 1) for i in range(T.length)],
                            'importance': [I for I in T.I_values]})
    t_gd = GridOptionsBuilder.from_dataframe(t_table)
    t_gd.configure_side_bar()
    t_gd.configure_selection(selection_mode='single', use_checkbox=True)
    st.subheader('Trajectory Table')
    st.caption('Importance is the difference between the maximum Q value and the minimum Q value of an action in '
               'a certain state.')
    t_grid_table = AgGrid(t_table, enable_enterprise_modules=True, height=175,
                          gridOptions=t_gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED,
                          fit_columns_on_grid_load=True)
    if t_grid_table["selected_rows"]:
        # show the related info of selected trajectory
        selection = int(t_grid_table["selected_rows"][0]['trajectory'].split(' ')[1]) - 1
        ms2_frags = st.session_state['store'][result]['ms2_frags']
        all = [e for e in env.vimms_env.mass_spec.fragmentation_events if e.ms_level == 2 or e.ms_level == 1]
        t_df = T.items[selection].get_df(max_peaks, feature_name)
        all_chemical = []
        for scanid in t_df['scan_id']:
            for e in all:
                if e.scan_id == scanid and e.chem not in all_chemical:
                    all_chemical.append(e.chem)
        # get query rt
        q_rt = []
        for scanid in t_df['scan_id']:
            flag = 0
            for e in all:
                if e.scan_id == scanid:
                    flag = 1
                    q_rt.append(round(e.query_rt, 4))
                    break
            if flag == 0:
                q_rt.append(None)
        # get chemicals
        chemical = []
        for scanid in t_df['scan_id']:
            for e in ms2_frags:
                if e.scan_id == scanid and e.chem not in chemical:
                    chemical.append(e.chem)
        # order chemicals by max intensity
        for i in range(len(chemical)):
            for j in range(len(chemical) - 1 - i):
                if chemical[j].max_intensity < chemical[j + 1].max_intensity:
                    chemical[j], chemical[j + 1] = chemical[j + 1], chemical[j]

        st.subheader('Detail info')
        st.caption('related info such features are listed blow')
        t_df.insert(loc=0, column='query_rt', value=q_rt)
        trajectory = GridOptionsBuilder.from_dataframe(t_df)
        trajectory.configure_side_bar()
        trajectory.configure_selection()
        AgGrid(t_df, enable_enterprise_modules=True, height=250,
               gridOptions=trajectory.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)

        st.subheader('Trajectory summary')
        st.markdown('**statistics**')
        st.caption('Coverage proportion measures how many chemicals a method is able to ‘hit’ with frag events.'
                   'Intensity proportion measures whether method can ‘hit’ with frag events at Max peak. Q-values ('
                   'action values) is an estimation of how good is it to take the action at the state.')
        # get coverage_p and intensity_p
        if len(chemical) != 0:
            # show info
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            coverage_p, intensity_p = get_proportion(ms2_frags, chemical)
            s_col1.metric('Coverage proportion', round(len(chemical) / len(all_chemical), 4))
            s_col2.metric('Intensity proportion', intensity_p)
            s_col3.metric('Max q-value', T.items[selection].maxq)
            s_col4.metric('Min q-value', T.items[selection].minq)
            s_col1.metric('Max reward', max(t_df['reward']))
            s_col2.metric('Min reward', min(t_df['reward']))
            s_col3.metric('Chemicals been fragmented', len(chemical))

        else:
            st.warning('No chemical is fragmented in this trajectory')
        # show plots
        st.markdown('**distributions of actions and rewards**')
        col11, col12 = st.columns(2)
        with col11:
            his_fig = px.histogram(t_df, x='action', nbins=max_peaks)
            st.plotly_chart(his_fig, use_container_width=True)
        with col12:
            his_fig = go.Figure()
            his_fig.add_trace(go.Histogram(x=t_df['reward'], xbins=dict(size=0.1), nbinsx=10))
            his_fig.update_layout(xaxis_title_text='reward', yaxis_title_text='count')
            st.plotly_chart(his_fig, use_container_width=True)

        st.subheader('View timestep')
        st.caption('These plots show the timesteps of the trajectory been selected. The start timestep used to form '
                   'the trajectory can be find on the plots.')
        col9, col10, col13 = st.columns(3)
        with col9:
            ac = t_df[t_df['timestep'] == T.items[selection].timestep]['action']
            fig = px.scatter(t_df, x='timestep', y='action')
            fig.add_annotation(x=T.items[selection].timestep,
                               y=ac[ac.index[0]],
                               text="start of trajectory",
                               showarrow=True,
                               arrowhead=1)
            st.plotly_chart(fig, use_container_width=True)
        with col10:
            re = t_df[t_df['timestep'] == T.items[selection].timestep]['reward']
            fig = px.scatter(t_df, x='timestep', y='reward')
            fig.add_annotation(x=T.items[selection].timestep,
                               y=re[re.index[0]],
                               text="start of trajectory",
                               showarrow=True,
                               arrowhead=1)
            st.plotly_chart(fig, use_container_width=True)
        with col13:
            im = t_df[t_df['timestep'] == T.items[selection].timestep]['importance']
            fig = px.scatter(t_df, x='timestep', y='importance')
            fig.add_annotation(x=T.items[selection].timestep,
                               y=im[im.index[0]],
                               text="start of trajectory",
                               showarrow=True,
                               arrowhead=1)
            st.plotly_chart(fig, use_container_width=True)

        if len(chemical) != 0:
            st.subheader('View fragmented chemicals in this trajectory')
            st.caption('All the chemicals been fragmented in this trajectory are shown in the following table, '
                       'select one of them to observe the chromatogram.')
            # make df
            chem_df = []
            for i in range(len(chemical)):
                store = pd.DataFrame(
                    {'index': [i + 1], 'mz': [round(chemical[i].mass, 4)], 'rt': [round(chemical[i].rt, 4)],
                     'max_intensity': [round(chemical[i].max_intensity, 4)]})
                chem_df.append(store)
            chem_df = pd.concat(chem_df)
            chem_gd = GridOptionsBuilder.from_dataframe(chem_df)
            chem_gd.configure_side_bar()
            chem_gd.configure_selection(selection_mode='single', use_checkbox=True)
            chem_grid_table = AgGrid(chem_df, enable_enterprise_modules=True, height=250,
                                     gridOptions=chem_gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED,
                                     fit_columns_on_grid_load=True)
            if chem_grid_table["selected_rows"]:
                st.caption('The blue line shown in the figure represents the chromatographic peak of the selected '
                           'chemical and the green dotted lines represent all MS2 events (fragmentation events) '
                           'occurring for the chemical.')
                selection = chem_grid_table["selected_rows"][0]['index']
                frags = [event for event in ms2_frags if chemical[selection - 1] == event.chem]
                # extract related RT and intensities
                chrom = chemical[selection - 1].chromatogram
                X = []
                Y = []
                rt = chrom.min_rt
                while rt <= chrom.max_rt:
                    X.append(rt + chemical[selection - 1].rt)
                    Y.append(chrom.get_relative_intensity(rt))
                    rt += (chrom.max_rt - chrom.min_rt) / 29
                # draw
                chem_fig = px.line(x=X, y=Y, labels={'x': 'RT', 'y': 'intensity'})
                for frag in frags:
                    chem_fig.add_vline(x=frag.query_rt, line_width=1, line_dash="dash", line_color="green")
                st.plotly_chart(chem_fig, use_container_width=True)


def get_proportion(ms2_frags, chems):
    all_intensity_props = []
    count = 0
    for chem in chems:
        events = [event for event in ms2_frags if chem == event.chem]
        chem_intensity_prop = 0.0
        if len(events) > 0:
            count += 1  # count how many chemicals are fragmented
            # find the intensities of the chemical at each ms2 event (when it gets fragmented)
            events_intensities = [event.parents_intensity[0] for event in events]
            # divide the largest intensity during fragmentation with the maximum intensity possible of this chemical
            chem_intensity_prop = max(events_intensities) / chem.max_intensity

        all_intensity_props.append(chem_intensity_prop)
    return round(count / len(chems), 4), round(sum(all_intensity_props) / len(all_intensity_props), 4)


main_window()
