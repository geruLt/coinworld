# coinworld
A reinforcement learning environment for cryptocoin trading

Currently there is only 5 coin available.
Since the environment works using real data, those data should be downloaded, and extracted to the raw_data folder.

Datas: https://drive.google.com/file/d/1IqF6R5qO1E7g-U-shT0gaTBYHVJZM0m7/view?usp=sharing

The environment is very similar to any other gym environment, with the basic reset, step functions. There is no render
function.

Environment can be tried with the run.py file, it uses the basic DQN agent (LSTM). 