# Deep-Variational-Reinforcement-Learning
This is the code accompanying the paper [Deep Variational Reinforcement Learning for POMDPs](https://arxiv.org/abs/1806.02426) by Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood and Shimon Whiteson.

Disclaimer: I cleaned up the code a bit before release. A few test runs indicate it still works but if you encounter problems please let me know, either here as issue or via email (maximilian.igl@gmail.com).
Also, if there are questions or something is unclear, please don't hesitate to approach me - feedback is very welcome!

# Running the code 

You can either use the provided docker container or install all dependencies.

## Using docker

With `nvidia-docker` installed, first create the container:
```
cd docker
./build.sh
```
which builds the docker container (this will take a few minutes). Once that is done, you can run experiments from the main folder in a container using 
```
cd ..
./docker/run.sh <gpu-nr> <name> <command>
```
for example
```
./docker/run.sh 0 test-run ./code/main.py -p with environment.config_file=openaiEnv.yaml
```
the results will be saved in the folder `saved_runs` using [this structure](https://sacred.readthedocs.io/en/latest/observers.html#file-observer). Please be sure to mount the folder accordingly if you want to access the data after the container finishes.

## Without docker

### Installing dependencies
You will need
- Python v3.6 (I used [Anaconda](https://conda.io/docs/user-guide/install/index.html) but it should work with other distributions as well)
- [Pytorch](https://pytorch.org/) v0.4.x
- [openai baselines](https://github.com/openai/baselines) (On MacOS, I needed to install mpi4py using conda beforehand to make the install not fail)
- `pip install 'gym[atari]'`

As well as other dependencies by running
```
pip install -r requirements.txt
```
in the main folder.

If you're running into an error with matplotlib on MacOS when running the RNN on MountainHike, you can use [this simple solution](https://stackoverflow.com/a/21789908/3730984).

### Running

From the main folder execute

```
python ./code/main.py -p with environment.config_file=openaiEnv.yaml
```
The results will be saved in the `saved_runs` folder in subfolders with incrementing numbers.

# Plotting

I included a very simple plotting script in the main folder:
```
python plot.py --id <id> [--metric <metric>]
```
where `<id>` is the ID of the experiment (created automatically and printed to command line when each run is started).
`<metric>` is which metric you want to plot. `result.true` is the default and probably what you want, i.e. the unclipped reward.

We use [sacred](https://github.com/IDSIA/sacred) for configuration and saving of results. It fully supports a more elaborat setup with SQL or noSQL databases in the background for storing and retrieving results. I stripped that functionality out for the release for ease of use but can highly recommend using it when working more extensively with the code.


# Reproducing results

Below are the commands to reproduce the results in the paper. Plots in the paper are averaged over 5 random seeds, but individual runs should qualitatively show the same results as training was fairly stable. If you run into problems, let me know (maximilian.igl@gmail.com).

## Default configuration

The default configuration can be found in `code/conf/` in the `default.yaml`. 
The environment must be specified in the command line by `environment.config_file='<envName>.yaml'`. The corresponding yaml file will be loaded as well (and overwrites some values in `default.yaml`, like for example the encoder/decoder architecture to match the observations space). 
Everything specified additionally in the command line overwrites the values in both yaml files.

DVRL:
```
python ./code/main.py -p with environment.config_file=openaiEnv.yaml environment.name=PongNoFrameskip-v0 algorithm.use_particle_filter=True algorithm.model.h_dim=256 algorithm.multiplier_backprop_length=10 algorithm.particle_filter.num_particles=15 opt.lr=2.0e-04 loss_function.encoding_loss_coef=0.1
```

RNN:
```
python ./code/main.py -p with environment.config_file=openaiEnv.yaml environment.name=PongNoFrameskip-v0 algorithm.use_particle_filter=False algorithm.model.h_dim=256 algorithm.multiplier_backprop_length=10  opt.lr=1.0e-04
```
(or with any other Atari environment)
**Please note that the results printed in the console are the _clipped_ rewards, for the true rewards please check 'result.true' in the metrics.json file or use the plotting script**

# Credits

The code is based on an older version of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr but heavily modified.


