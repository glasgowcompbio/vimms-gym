# vimms-gym

The choice of fragmentation strategy used for data acquisition in untargeted metabolomics greatly
affects the coverage and spectral quality of identified metabolites. In a typical strategy for
data-dependant acquisition, the most *N* intense ions (Top-N) in the survey MS1 scan are chosen for
fragmentation. This strategy is not entirely data-driven as it is unable to learn and adapt to changes 
in incoming signals when deciding which ions to target for fragmentation.

[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) has been widely used
to train intelligent agents that manage the operations of various scientific
instruments ([example](https://www.nature.com/articles/s41586-021-04301-9)). However, its use in
mass spectrometry instrumentation control and untargeted metabolomics has never been explored.

In vimms-gym, we provide an [OpenAI gym](https://gym.openai.com/)
environment to develop data-dependant fragmentation strategies that control a simulated mass
spectrometry instrument and learn from the data during acquisition. This is built
upon [ViMMS](https://github.com/glasgowcompbio/vimms), a general framework to develop, test and
optimise fragmentation strategies. We hope that vimms-gym could encourage further research into
applying reinforcement learning in data acquisition for untargeted metabolomics.

## Installation

No Python package is provided at the moment as the project is still under active development.

To use vimms-gym, please clone this repository first, then use your preferred method to install the
required dependencies.

***A. Managing Dependencies using Pipenv***

1. Install pipenv (https://pipenv.readthedocs.io).
2. In the cloned Github repo, run `$ pipenv install` to create a new virtual environment and
   install all the packages need to run ViMMS.
3. Go into the newly created virtual environment in step (4) by typing `$ pipenv shell`.
4. In this environment, you could develop run the environment, train models etc by running
   notebooks (`$ jupyter lab`).

***B. Managing Dependencies using Pipenv***

1. Install Anaconda Python (https://www.anaconda.com/products/individual).
2. In the cloned Github repo, run `$ conda env create --file environment.yml` to create a new
   virtual environment and install all the packages need to run ViMMS.
3. Go into the newly created virtual environment in step (4) by typing `$ conda activate vimms-gym`
   .
4. In this environment, you could develop run the environment, train models etc by running
   notebooks (`$ jupyter lab`).

The [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) package has been
included as a dependency of this project, although you may use other RL frameworks to work with
vimms-gym if desired.

## Examples

Example notebooks can be found [here](https://github.com/glasgowcompbio/vimms-gym/tree/main/notebooks).
This includes a demonstration of the environment, as well as other notebooks to train models and
evaluate the results.
