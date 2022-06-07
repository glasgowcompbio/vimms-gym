#!/usr/bin/env python
import sys

sys.path.append('..')

import streamlit as st

from vimms_gym.common import METHOD_PPO, METHOD_TOPN, METHOD_RANDOM, METHOD_FULLSCAN, render_scan
from vimms_gym.viewer_helper import run_simulation, preset_1, preset_2, \
    load_model_and_params
from vimms_gym.chemicals import generate_chemicals


def main():
    st.title('ViMMS-Gym Viewer 👀')
    st.subheader('Interactive visualisation of pre-trained ViMMS-Gym model')
    method, params = render_sidebar()

    st.sidebar.markdown("""---""")
    if st.sidebar.button('Run episode'):  # the Run button is clicked

        # if no cache episode exists and a valid preset has been selected,
        # generate a new episode
        if 'episode' not in st.session_state and params is not None:
            episode = generate_episode(method, params)

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

    st.sidebar.write('Select a preset environment or upload your own chemicals')
    choices = [
        'Simulated chems',
        'QCB chems'
    ]
    preset = st.sidebar.radio('Select a preset environment', choices)
    if preset == 'Simulated chems':
        st.sidebar.caption('Generate 2000 - 5000 chemical objects with uniform distributions.')
        params = preset_1()

    elif preset == 'QCB chems':
        st.sidebar.caption(
            'Generate 2000 - 5000 chemical objects with more realistic distributions.')
        st.write('This preset is not implemented yet')
        params = preset_2()

    model_file = st.sidebar.file_uploader('Or, upload ViMMS chemicals')

    st.sidebar.markdown("""---""")
    st.sidebar.header('Define fragmentation policy')
    choices = (METHOD_PPO, METHOD_TOPN, METHOD_RANDOM, METHOD_FULLSCAN)
    method = st.sidebar.radio('Select a policy', choices)
    if method == METHOD_PPO:  # TODO: unused
        model_file = st.sidebar.file_uploader('Upload pre-trained PPO model (StableBaselines3)')

    return method, params


def generate_episode(method, params):
    if params is not None:  # and a valid preset has been selected

        max_peaks = 200  # TODO: should be part of the preset

        # generate chemicals following the selected preset
        chemical_creator_params = params['chemical_creator']
        chems = generate_chemicals(chemical_creator_params)
        st.metric(label='Chemicals', value=len(chems))

        # run simulation to generate an episode
        N, min_ms1_intensity, model, params = load_model_and_params(method, params)
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

        col1, col2 =  st.columns([0.2, 1])
        with col1:
            if st.button('< Previous'):
                if st.session_state['step'] > 0:
                    st.session_state['step'] -= 1
        with col2:
            if st.button('Next >'):
                if st.session_state['step'] < episode.num_steps-1:
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
            prev_step_data = episode.get_step_data(step-1)
            prev_reward = prev_step_data['reward']
            delta = reward - prev_reward
            st.metric(label='Reward', value='%.3f' % reward, delta='%.3f' % delta)
        with col4:
            ms_level = state['ms_level'][0]
            st.metric(label='MS level', value=int(ms_level))

        # render current and last ms1 scans from the info field
        info = step_data['info']
        col1, col2 = st.columns(2)
        with col1:
            current_scan = info['current_scan']
            fig = render_scan(current_scan)
            if fig is not None:
                st.pyplot(fig)
                st.caption('Current scan')
        with col2:
            last_ms1_scan = info['last_ms1_scan']
            fig = render_scan(last_ms1_scan)
            if fig is not None:
                st.pyplot(fig)
                st.caption('Last MS1 scan')

        st.subheader('Step data')
        st.json(step_data)


# pandas display options
main()
