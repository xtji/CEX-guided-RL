# CEX-guided-RL

This project implements Counterexample-guided RL, an online-offline RL method that improves safety during online exploration by leveraging counterexample submodels during offline training, as described in the paper "Probabilistic Counterexample Guidance for Safer Reinforcement Learning": https://arxiv.org/abs/2307.04927

## Installation
### Docker Image Usage
To simplify the installation of the optimisers, we provide a Dockerfile for this demo. If you need to install Docker Engine, see the installation guide available at: https://docs.docker.com/engine/install/.

Before building the Docker image, please download the scipoptsuite from https://scipopt.org/download.php?fname=scipoptsuite-8.0.3.tgz and put it in the same folder as the Dockerfile in order to install the SCIP optimiser.

* You can build and run the Docker image using the following commands:

`docker build -t cex-guided-rl .`

`docker run -it cex-guided-rl`

### Local Usage

* If you prefer to run this demo without Docker, the Python dependencies(compatible with Python 3.9 through 3.11) can be installed using these commands:

`pip install --no-cache-dir --no-dependencies -r requirements.txt`

`pip install torch`

* We evaluate this method in four gym-like environments. 
Custom gym environments can be installed with the following commands:

`pip install -e gym-gridworld/ `

`pip install -e gym-marsrover/ `

"FrozenLake8x8-v1" can be installed with `openai-gym` automatically, and the environmental setting details of FrozenLake can be found at: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py


## Training
To train the agent with CEX-guided RL, run the following command:

`python training.py --config-name=config_name`

where `config_name` is the name of the configuration file in the `conf/` folder. For example, to train the agent in the HybridGrid environment, run:

`python training.py --config-name=hybridgrid`

### Configuration
* We provide training hyperparameters for four environments in the `conf/` folder: 
  `frozenlake8x8.yaml, discretegrid.yaml, hybridgrid.yaml, marsrover.yaml`.

* Specifically, `frozenlake8x8.yaml` and `discregrid.yaml` used for training the QL-agent in the slippery FrozenLake8x8 and DiscreteGrid environments, meanwhile, `hybridgrid.yaml` and `marsrover.yaml` are used for training the DQN-agent in HybridGrid and MarosRover nvironments under counterexample guidance.


* To train the agent using the baseline QL and DQN methods instead, please set `alg.guidance` to `False` in corresponding configuration files.

### Optimiser
In the paper, we used the Gurobi optimiser (v9.1.0). However, since Gurobi requires installation and a license, we offer
`SCIP` and `GLPK-MI` as alternative optimisers in the Docker image. Please note that these alternatives may take significantly longer for optimisation in the FrozenLake8x8 and MarsRover environments.

For better performance with the Gurobi optimiser (v9.1.0 was used in our evaluation), see the installation guide at: https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-. Also, instructions for acquiring a free academic Web License Service (WLS) for Docker can be found at: https://www.gurobi.com/features/web-license-service/.











