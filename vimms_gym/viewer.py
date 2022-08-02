#!/usr/bin/env python
import sys

sys.path.append('..')

import streamlit as st

from vimms_gym.common import METHOD_PPO, METHOD_TOPN, METHOD_RANDOM, METHOD_FULLSCAN, render_scan, \
    METHOD_DQN
from vimms_gym.viewer_helper import run_simulation, load_model_and_params, get_parameters, \
    scan_id_to_scan
from vimms_gym.chemicals import generate_chemicals
from vimms_gym.experiments import preset_qcb_small, preset_qcb_medium, preset_qcb_large


def main():
    st.title('ViMMS-Gym Viewer 👀')
    st.subheader('Interactive visualisation of pre-trained ViMMS-Gym model')
    preset, method = render_sidebar()

    st.sidebar.markdown("""---""")
    if st.sidebar.button('Run episode'):  # the Run button is clicked

        # if no cache episode exists, generate a new episode
        if 'episode' not in st.session_state:
            episode = generate_episode(preset, method)
            if episode is not None:
                # store episode into streamlit's session state to prevent repeated computation
                st.session_state['episode'] = episode
                st.session_state['step'] = 0
                # st.write('Episode stored into session')

    st.sidebar.write(' ')
    st.sidebar.write(' ')

    # main visualisation loop starts here
    handle_slider()


def render_sidebar():
    st.sidebar.title('Simulation Parameters')

    st.sidebar.header('Generate input chemicals')
    params = None
    max_peaks = None

    choices = [
        'QCB_chems_small',
        'QCB_chems_medium',
        'QCB_chems_large',
    ]
    preset = st.sidebar.radio('Select a preset environment', choices)
    if preset == 'QCB_chems_small':
        st.sidebar.caption('Generate 20 - 50 chemical objects that resembles the QC Beer sample.')
    if preset == 'QCB_chems_medium':
        st.sidebar.caption(
            'Generate 200 - 500 chemical objects that resembles the QC Beer sample.')
    elif preset == 'QCB_chems_large':
        st.sidebar.caption(
            'Generate 2000 - 5000 chemical objects that resembles the QC Beer sample.')
        st.sidebar.warning('This preset is not implemented yet')

    st.sidebar.markdown("""---""")
    st.sidebar.header('Define fragmentation policy')
    choices = (METHOD_PPO, METHOD_DQN, METHOD_TOPN, METHOD_RANDOM, METHOD_FULLSCAN)
    method = st.sidebar.radio('Select a policy', choices)
    return preset, method


def generate_episode(preset, method):
    params, max_peaks = get_parameters(preset)
    episode = None
    if params is not None:
        # generate chemicals following the selected preset
        chemical_creator_params = params['chemical_creator']
        chems = generate_chemicals(chemical_creator_params)
        st.metric(label='Chemicals', value=len(chems))

        # run simulation to generate an episode
        N, min_ms1_intensity, model, params = load_model_and_params(preset, method, params)
        episode = run_simulation(N, chems, max_peaks, method, min_ms1_intensity, model,
                                 params)
    return episode


def handle_slider():
    if 'episode' in st.session_state:
        episode = st.session_state['episode']
        st.subheader('Explore episode')

        # https://discuss.streamlit.io/t/update-slider-value/372/2
        # The slider needs to come after the button, to make sure the first increment
        # works correctly. So we create a placeholder for it here first, and fill it in
        # later.
        widget = st.empty()

        col1, col2 = st.columns([0.2, 1])
        with col1:
            if st.button('< Previous'):
                if st.session_state['step'] > 0:
                    st.session_state['step'] -= 1
        with col2:
            if st.button('Next >'):
                if st.session_state['step'] < episode.num_steps - 1:
                    st.session_state['step'] += 1

        step = widget.slider('Select a timestep', min_value=0, max_value=episode.num_steps,
                             value=st.session_state['step'], step=1)
        st.session_state['step'] = step
        step_data = episode.get_step_data(step)

        # display some metrics from state
        state = step_data['state']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label='Timestep', value=step)
        with col2:
            action = step_data['action']
            st.metric(label='Action', value=action)
        with col3:
            reward = step_data['reward']
            prev_step_data = episode.get_step_data(step - 1)
            prev_reward = prev_step_data['reward']
            delta = reward - prev_reward
            st.metric(label='Reward', value='%.3f' % reward, delta='%.3f' % delta)
        with col4:
            ms_level = state['ms_level'][0]
            st.metric(label='MS level', value=int(ms_level))

        # render current and last ms1 scans from the info field
        info = step_data['info']

        # NOTE: this no longer works because it made training much slower!!
        # Now we only return the current_scan_id in the info field, so we need
        # to use this to look up the actual scan object.
        # current_scan = info['current_scan']
        current_scan_id = info['current_scan_id']
        current_scan = scan_id_to_scan(episode.scans, current_scan_id)

        fig = render_scan(current_scan)
        if fig is not None:
            st.pyplot(fig)
            st.caption('Current scan')

        st.subheader('Step data')
        st.json(step_data)


# pandas display options
main()
