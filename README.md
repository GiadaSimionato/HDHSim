# HDHSim: A Simulator for Hole Detection and Healing 

This repository contains HDHSim: a discrete-time Python simulator for Hole Detection and Healing purposes.
By default, HDHSim runs the algorithm described in Simionato G. et al., [2023] (https://doi.org/10.1145/3583131.3590468).

## Contents:

- [Installation](#installation)
- [Execution](#execution)
- [Implementing your algorithm](#implementing-your-algorithm)


## Installation
1. Clone the repository
```
git clone git@github.com:GiadaSimionato/HDHSim.git
```
2. Install the requirements
```
pip install -r requirements.txt
```
3. To log the experiments, HDHSim uses [ClearML](https://clear.ml/). Please, follow the instructions at https://shorturl.at/wxEYZ to install and setup ClearML Logger.


## Execution

1. To handle the parameters configuration, HDHSim uses [Hydra](https://github.com/facebookresearch/hydra). Therefore, to set the desired parameters either modify `config/config.yaml` or create a new `.yaml` file.
2. Run `python main.py` to execute the algorithm.
3. To download the experiments from ClearML, use `tools/download-results.py` specifying tag and credentials.


## Implementing your algorithm

### How to costumize the scenario

To customize the scenario, set the number of nodes of the original network and their sensing range at `config.yaml/network/n_nodes` and `config.yaml/network/rs_nodes`. The number and size of holes must be set using `config.yaml/scenario/ntb`. E.g. ntb = [3,5,4] represents three holes in the network caused by 3, 5 and 4 faulty nodes.
The number of release points (RPs) must be specified at `config.yaml/init/agents/nests` while the number of agents per RP at `config.yaml/init/agents/nest/agents`.

### How to costumize the logic

You can write the logic that will be executed at each tick of the simulation for both nodes and agents.
To specify the node logic, fill `node_logic` in `src/logic/controller/core.py`. For the agent logic, fill `agent_logic` in the same file.
To add attributes to the node or agent objects, modify `init_nodes` or `init_agents` in `src/logic/controller/init.py`, respectively.
To change the rendering of nodes and agents, modify `src/logic/controller/render.py`.

### How to costumize the evaluation metrics and add hooks

You can change the evaluation metrics by modifying `src/logic/controller/metrics.py`. The content of the `start` function is executed once at the beginning of the simulation; `tick` is executed at each tick, while `end` is executed once at the end of the simulation.

It is possible to add external functions executed pre or post tick by extending the Hook class. An example is the `Msg` class in `src/logic/controller/core.py`, that allows exchanging messages among nodes and agents (specifying the ids of the sender/receiver and the message through a dictionary). 


For any question about the simulator, please write to giada.simionato@phd.unipi.it or federico.galatolo@unipi.it.
