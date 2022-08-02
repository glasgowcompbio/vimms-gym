This folder contains notebooks used to train ViMMS-Gym using stable-baselines3 on a variety of
data. It also includes notebooks to evaluate the results and visualise the actions that a
pre-trained agent takes. Here's a brief explanation of the folder structure and 
files inside.

### 1. simulated_chems

a. **training.ipynb**: train ViMMS-Gym using easy synthetic chemicals, where
   m/z, RT and intensity distributions of chemicals are all uniform. Gaussian shapes 
   are assumed for the chromatograms.

b. **evaluation.ipynb**: evaluate the performance of pre-trained model from (1a) above.

c. **visualisation.ipynb**: make various plots to visualise the result of pre-trained model 
   from (1b) above.

### 2. QCB_chems

a. **training.ipynb**: train ViMMS-Gym using synthetic chemicals that are more realistic. 
   The generated chemicals should have characteristics such as m/z, RT, intensity distributions,
   as well as chromatographic peak shapes that resemble that of a real experimental QC Beer data.

b. **evaluation.ipynb**: evaluate the performance of pre-trained model from () above.

c. **visualisation.ipynb**: make various plots to visualise the result of pre-trained model 
   from (2b) above.

### 3. actual_QCB

a. **evaluation.ipynb**: evaluate the performance of pre-trained model from (2a) above on
   actual chemicals extracted from the QC Beer data against peak picking results.

b. **visualisation.ipynb**: make various plots to visualise the result of pre-trained model 
   from (3a) above.

