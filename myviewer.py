"""
Created on 16/06/2022 20:42
@author: Liu Ziyan
@E-mail: 2650906L@student.gla.ac.uk
"""
import streamlit as st
import xgboost as xgb
import shap
import time
from vimms_gym.env import DDAEnv
from vimms_gym.common import METHOD_PPO, METHOD_TOPN, METHOD_RANDOM, METHOD_FULLSCAN
from vimms_gym.common import MAX_OBSERVED_LOG_INTENSITY, MAX_ROI_LENGTH_SECONDS
from vimms_gym.viewer_helper import run_simulation, preset_1, preset_2, \
    load_model_and_params
from vimms_gym.chemicals import generate_chemicals
import plotly.express as px
import pandas as pd
import numpy as np


def main_window():
    st.title('VIMMS-Gym Viewer')
    st.header("Interactive visualisation of VIMMS-Gym model")
    params = sidebar_environment()  # sidebar
    if st.sidebar.button('Generate chemicals'):  # button
        # store
        st.session_state['chems'] = Generate_chems(params)
        st.session_state['results'] = {}
        st.session_state['store'] = {}

    policy = sidebar_policy()  # sidebar

    if st.sidebar.button('Run episode'):  # button
        info_container = st.empty()
        with info_container.container():
            episode, max_peaks, env = train(policy, params, st.session_state['chems'])
            # store
            st.session_state['results'][policy] = [episode, env, max_peaks]
            st.session_state['store'][policy] = {}
        info_container.empty()
    st.sidebar.write('')
    st.sidebar.write('')
    visual_window()


def sidebar_environment():
    st.sidebar.header("Simulation Parameters")
    st.sidebar.subheader('Generate chemicals')
    st.sidebar.write('Select a preset environment or upload your own chemicals')
    params = None
    # env selection
    option = st.sidebar.selectbox(
        'Select environment',
        ('Existing preset environment', 'Upload file'))
    if option == "Existing preset environment":
        environment = st.sidebar.radio(
            "Select a preset environment",
            ('Simulated chemicals', 'QCB chemicals'))
        if environment == "Simulated chemicals":
            st.sidebar.caption('Generate 2000 - 5000 chemical objects with uniform distributions.')
            params = preset_1()
        elif environment == "QCB chemicals":
            st.sidebar.caption('Generate 2000 - 5000 chemical objects with more realistic distributions.')
            st.warning('This preset is not implemented yet')
            params = preset_2()
    elif option == "Upload file":
        environment = st.sidebar.file_uploader("Upload ViMMS chemicals")
        params = None
        st.warning('This preset is not implemented yet')
    return params


def sidebar_policy():
    st.sidebar.markdown("---")
    # policy selection
    st.sidebar.subheader("Fragmentation policy")
    policy = st.sidebar.radio(
        "Select a policy",
        (METHOD_TOPN, METHOD_PPO))

    # if policy == METHOD_PPO:
    # model_file = st.sidebar.file_uploader('Upload pre-trained PPO model (StableBaselines3)')
    st.sidebar.markdown("---")
    return policy


def Generate_chems(params):
    if params is not None:
        # generate chemicals following the selected preset
        chemical_creator_params = params['chemical_creator']
        chems = generate_chemicals(chemical_creator_params)
        info_container = st.empty()
        with info_container.container():
            st.metric(label='', value="{0} chemicals have been generated".format(str(len(chems))))
            time.sleep(1.5)
        info_container.empty()
        return chems
    else:
        st.error("parameters error!")


def train(policy, params, chems):  # trainning function
    if params is not None:
        max_peaks = 200
        # run simulation to generate an episode
        N, min_ms1_intensity, model, params = load_model_and_params(policy, params)
        episode, env = run_simulation(N, chems, max_peaks, policy, min_ms1_intensity, model, params)
        return episode, max_peaks, env
    else:
        st.error("parameters error!")


def visual_window():  # interpretation part
    if 'results' in st.session_state and st.session_state['results'] != {}:
        st.header('Explore episode')
        # set up the placeholder
        explore_window = st.empty()
        plots_window = st.empty()
        # get data which been stored
        result = explore_window.selectbox('Select results', st.session_state['results'].keys())
        episode = st.session_state['results'][result][0]
        env = st.session_state['results'][result][1]
        max_peaks = st.session_state['results'][result][2]
        # methods selection
        vm = plots_window.selectbox('Select visualization methods',
                                    ('view features', 'view chemicals', 'view trajectory'))
        st.markdown("---")
        if vm == 'view features':
            st.header('Feature + Action + Rewards')
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
                # extract feature data
                for step in range(episode.num_steps):
                    if episode.actions[step] != max_peaks and step < episode.num_steps - 1:
                        intensities.append(
                            episode.observations[step + 1]['intensities'][
                                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY)
                        excluded.append(episode.observations[step + 1]['excluded'][episode.actions[step]] * env.rt_tol)
                        roi_length.append(
                            episode.observations[step + 1]['roi_length'][
                                episode.actions[step]] * MAX_ROI_LENGTH_SECONDS)
                        roi_elapsed_time_since_last_frag.append(
                            episode.observations[step + 1]['roi_elapsed_time_since_last_frag'][
                                episode.actions[step]] * MAX_ROI_LENGTH_SECONDS)
                        roi_intensity_at_last_frag.append(
                            episode.observations[step + 1]['roi_intensity_at_last_frag'][
                                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY
                        )
                        roi_min_intensity_since_last_frag.append(
                            episode.observations[step + 1]['roi_min_intensity_since_last_frag'][
                                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY
                        )
                        roi_max_intensity_since_last_frag.append(
                            episode.observations[step + 1]['roi_max_intensity_since_last_frag'][
                                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY
                        )
                        ac.append(episode.actions[step])
                        re.append(episode.rewards[step])
                        index.append(step)
                # make dataframe
                df = pd.DataFrame(
                    {'timestep': index, 'action': ac, 'reward': re, 'intensities': intensities, 'excluded': excluded,
                     'roi_length': roi_length, 'roi_elapsed_time_since_last_frag': roi_elapsed_time_since_last_frag,
                     'roi_intensity_at_last_frag': roi_intensity_at_last_frag,
                     'roi_min_intensity_since_last_frag': roi_min_intensity_since_last_frag,
                     'roi_max_intensity_since_last_frag': roi_max_intensity_since_last_frag})
                st.session_state['store'][result]['feature'] = df

            if 'model' not in st.session_state['store'][result]:
                st.session_state['store'][result]['model'] = {}
                # colnames which will be used later
                cols = ['intensities', 'excluded', 'roi_length', 'roi_elapsed_time_since_last_frag',
                        'roi_intensity_at_last_frag',
                        'roi_min_intensity_since_last_frag', 'roi_max_intensity_since_last_frag']
                # first build regression model
                ac_xgbmodel = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
                ac_xgbmodel.fit(st.session_state['store'][result]['feature'][cols],
                                st.session_state['store'][result]['feature']['action'].values)

                re_xgbmodel = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
                re_xgbmodel.fit(st.session_state['store'][result]['feature'][cols],
                                st.session_state['store'][result]['feature']['reward'].values)
                st.session_state['store'][result]['model']['action'] = ac_xgbmodel
                st.session_state['store'][result]['model']['reward'] = re_xgbmodel

            view_feature(episode, result)

        if vm == 'view chemicals':
            st.header('Top intensity chemicals')
            amount = st.radio('select Top chemicals', ('Top 50', 'Top 100', 'Top 200'), horizontal=True)
            if 'sorted_chemicals' not in st.session_state['store'][result]:
                sorted_chemicals = st.session_state['chems']
                for i in range(len(sorted_chemicals)):
                    for j in range(len(sorted_chemicals) - 1 - i):
                        if sorted_chemicals[j].max_intensity < sorted_chemicals[j + 1].max_intensity:
                            sorted_chemicals[j], sorted_chemicals[j + 1] = sorted_chemicals[j + 1], sorted_chemicals[j]
                st.session_state['store'][result]['sorted_chemicals'] = sorted_chemicals

            view_chemical(env, amount, result)

        if vm == 'view trajectory':
            pass


def view_feature(episode, result):
    # draw SHAP plots
    st.subheader('SHAP Plot--feature importance')
    Y = st.radio('select Action or Reward', ('action', 'reward'), horizontal=True)
    # remove warning of st
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # use SHAP to explain
    explainer = shap.TreeExplainer(st.session_state['store'][result]['model'][Y])
    cols = ['intensities', 'excluded', 'roi_length', 'roi_elapsed_time_since_last_frag',
            'roi_intensity_at_last_frag',
            'roi_min_intensity_since_last_frag', 'roi_max_intensity_since_last_frag']
    shap_values = explainer.shap_values(st.session_state['store'][result]['feature'][cols])
    # show plots in two columns
    col1, col2 = st.columns(2)
    with col1:
        # draw
        shap_plot = shap.summary_plot(shap_values, st.session_state['store'][result]['feature'][cols])
        st.pyplot(shap_plot, use_container_width=True)
    with col2:
        # draw
        shap_plot = shap.summary_plot(shap_values, st.session_state['store'][result]['feature'][cols], plot_type='bar')
        st.pyplot(shap_plot, use_container_width=True)

    st.markdown("---")
    st.subheader('View features')
    feature = st.radio('select a feature', ('intensities', 'excluded', 'roi_length', 'roi_elapsed_time_since_last_frag',
                                            'roi_intensity_at_last_frag', 'roi_min_intensity_since_last_frag',
                                            'roi_max_intensity_since_last_frag'), horizontal=True)
    # df is reconstructed according to the selected feature to realize slider-based control of the image
    feature_vector = st.session_state['store'][result]['feature'][feature]
    feature_range = st.slider('select feature range',
                              round(min(feature_vector), 2), round(max(feature_vector), 2),
                              (round(min(feature_vector), 2), round(max(feature_vector), 2)))
    # distinguish the color
    new_feature_vector = []
    for value in feature_vector:
        if feature_range[0] <= value <= feature_range[1]:
            new_feature_vector.append("in range")
        else:
            new_feature_vector.append("out range")
    plot_df = pd.DataFrame({'timestep': st.session_state['store'][result]['feature']['timestep'],
                            'action': st.session_state['store'][result]['feature']['action'],
                            'reward': st.session_state['store'][result]['feature']['reward'],
                            'feature': new_feature_vector})

    timestep_range = st.slider('select timestep range',
                               0, episode.num_steps, (0, episode.num_steps))
    # reset the axis range
    plot_df = plot_df[plot_df['timestep'] > timestep_range[0]]
    plot_df = plot_df[plot_df['timestep'] < timestep_range[1]]
    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(plot_df, x='timestep', y='action', color='feature')
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.scatter(plot_df, x='timestep', y='reward', color='feature')
        st.plotly_chart(fig, use_container_width=True)


def view_chemical(env, amount, result):
    # get fragment info
    ms2_frags = [e for e in env.vimms_env.mass_spec.fragmentation_events if e.ms_level == 2]
    # slider
    amount_slider = st.slider('select a chemical', 1, int(amount.split(' ')[1]))
    # get chemicals
    chemical = st.session_state['store'][result]['sorted_chemicals'][0:int(amount.split(' ')[1])]
    # get related ms2 events
    frags = [event for event in ms2_frags if chemical[amount_slider - 1] == event.chem]
    # extract related RT and intensities
    chrom = chemical[amount_slider - 1].chromatogram
    X = []
    Y = []
    rt = chrom.min_rt
    while rt <= chrom.max_rt:
        X.append(rt + chemical[amount_slider - 1].rt)
        Y.append(chrom.get_relative_intensity(rt))
        rt += (chrom.max_rt - chrom.min_rt) / 19
    # draw plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('View chemicals')
        fig = px.line(x=X, y=Y, labels={'x': 'RT', 'y': 'intensity'})
        if frags:
            for frag in frags:
                fig.add_vline(x=frag.query_rt, line_width=1, line_dash="dash", line_color="green")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader('Related info')
        st.write("chemical info")
        st.write(chemical[amount_slider - 1])
        st.write("MS2 events info")
        if frags:
            for frag in frags:
                st.write(frag)


main_window()
