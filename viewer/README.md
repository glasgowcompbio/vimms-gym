## Introduction

This project aims to try to help user understand vimms-gym models through a visual app. This project has completed an interactive web application. In the application, users can complete the training simulation of vimms-gym model and the visualization of the results. The data used in the present simulations were obtained from the BEER sample of Glasgow Polyomics. In the app, users can observe different features, how chemical are fragmented, and trajectory analysis for one episode.

## Install vimms-gym

To use this app, you need to install vimms-gym first. Currently, vimms-gym is not in the form of package and is in the development stage. Please clone the contents of the entire viewer branch repository first.

***install dependencies through Anaconda***

1. Install Anaconda (https://www.anaconda.com/products/individual). If you are using windows, please select the `Add Anaconda to my PATH environment variable` in `Advance Options`.
2. Clone the the viewer branch.
3. Go to the location of the cloned repository in the terminal and run `$ conda env create --file environment.yml` to create a new virtual environment of vimms-gym. This may take a while.

## Usage

1. Go into the virtual environment of vimms-gym by typing `$ conda activate vimms-gym` in terminal.
2. Go into the cloned repository through terminal.
3. Go into viewer folder by typing `$ cd viewer`.
4. Run the app by typing `$ streamlit run myviewer.py`.
5. In the app, please generate chemical first. Each policy can only run once for each generated chemical set. If you want test more please generate new chemical set.

## Files

1. **myviewer.py**: the main file of the app. This file finishes all the functions of app. The contents of the documents are all completed by Ziyan.
2. **viewer_helper.py**: help codes for running simulation. Functions such as get_parameters(), load_model_and_params(), scan_id_to_scan() and part of run_simulation() are provided by Joe Wandy. Trajectory() object, PriorityQueue() object and HIGHLIGHTS algorithm in run_simulation() are copmleted by Ziyan.
3. **experiments.py**: parameters of three preset environments for beer samples. This file helps to finish the function of extract and store the simulated chemicals. The whole file are provided by Joe Wandy.
4. **.p files**: data of simulated chemicals extracted in 3.

