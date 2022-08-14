## Introduction

This project aims to try to help user understand vimms-gym models through a visual app. This project has completed an interactive web application. In the application, users can complete the training simulation of vimms-gym model and the visualization of the results. The data used in the present simulations were obtained from the BEER sample of Glasgow Polyomics. In the app, users can observe different features, how chemical are fragmented, and trajectory analysis for one episode.

## Install vimms-gym

To use this app, you need to install vimms-gym first. Currently, vimms-gym is not in the form of package and is in the development stage. Please clone the contents of the entire branch repository first.

***install dependencies through Anaconda***

1. Install Anaconda (https://www.anaconda.com/products/individual).
2. Cloned the whole repository.
3. Go to the location of the clone repository in the terminal and run `$ conda env create --file environment.yml` to create a new virtual environment of vimms-gym.

## Usage

1. Go into the virtual environment of vimms-gym by typing `$ conda activate vimms-gym` in terminal.
2. Go into the cloned repository through terminal.
3. Go into viewer folder by typing `$ cd viewer`
2. Run the app by typing `$ streamlit run myviewer.py`.



