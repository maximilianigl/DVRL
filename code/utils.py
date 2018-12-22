import torch
import numpy as np
from torch.autograd import Variable
import errno
import os
import json
import sys
from glob import glob
import os.path as osp
import pymongo
import logging
import aesmc.math as math
import collections
from docopt import docopt
from sacred.observers import MongoObserver
from sacred.arg_parser import get_config_updates


def safe_make_dirs(path):
    """
    Given a path, makes a directory. Doesn't make directory if it already exists. Treats possible
    race conditions safely.
    http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def toOneHot(action_space, actions):
    """
    If action_space is "Discrete", return a one hot vector, otherwise just return the same `actions` vector.

    actions: [batch_size, 1] or [batch_size, n, 1]

    If action space is continuous, just return the same action vector.
    """
    # One hot encoding buffer that you create out of the loop and just keep reusing
    if action_space.__class__.__name__ == "Discrete":
        nr_actions = action_space.n
        actions_onehot_dim = list(actions.size())
        actions_onehot_dim[-1] = nr_actions

        actions = actions.view(-1, 1).long()
        action_onehot = torch.FloatTensor(actions.size(0), nr_actions)

        return_variable = False
        if isinstance(actions, Variable):
            actions = actions.data
            return_variable = True

        # In your for loop
        action_onehot.zero_()
        if actions.is_cuda:
            action_onehot = action_onehot.cuda()
        action_onehot.scatter_(1, actions, 1)

        if return_variable:
            action_onehot = Variable(action_onehot)

        action_onehot.view(*actions_onehot_dim)

        return action_onehot
    else:
        return actions.detach()


def save_model(dir, name, model, _run):
    """
    Save the model to the observer using the `name`.
    _run is the _run object from sacred.
    """

    name_model = dir + '/' + name
    torch.save(model.state_dict(), name_model)

    s_current = os.path.getsize(name_model) / (1024 * 1024)

    _run.add_artifact(name_model)
    os.remove(name_model)

    logging.info('Saving model {}: Size: {} MB'.format(name, s_current))


def save_numpy(dir, name, array, _run):
    """
    Save a numpy array to the observer, using the `name`.
    _run is the _run object from sacred.
    """

    name = dir + '/' + name
    np.save(name, array.astype(np.float32))
    s_current = os.path.getsize(name) / (1024 * 1024)
    _run.add_artifact(name)
    os.remove(name)
    logging.info('Saving observations {}: Size: {} MB'.format(name, 2 * s_current))


def load_results(dir):
    """
    Since we are using clipped rewards (e.g. in Atari games), we need to access the monitor
    log files to get the true returns.

    Args:
        dir: Directory of the monitor files

    Returns:
        df: A pandas dataframe. Forgot the dimensions but it works with the function `log_and_print`
    """
    import pandas
    monitor_files = (glob(osp.join(dir, "*monitor.csv")))
    if not monitor_files:
        raise Exception("no monitor files of the found")
    dfs = []
    headers = []
    for run_nr, fname in enumerate(monitor_files):
        with open(fname, 'rt') as fh:
            firstline = fh.readline()
            assert firstline[0] == '#'
            header = json.loads(firstline[1:])
            df = pandas.read_csv(fh, index_col=None)
            headers.append(header)
            df['t'] += header['t_start']
            df['run_nr'] = run_nr
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df['headers'] = headers  # HACK to preserve backwards compatibility
    return df


def print_header():
    logging.info('      Progr | FPS | NKP | ToL | avg | med | min | max || Losses: | ent | val | act | enc || pri | emi | rew |')
    logging.info('      ------|-----|-----|-----|-----|-----|-----|-----||---------|-----|-----|-----|-----||-----|-----|-----|')


def log_and_print(j, num_updates, T, id_tmp_dir, final_rewards, tracking,
                  num_ended_episodes, avg_nr_observed, avg_encoding_loss,
                  total_loss, value_loss, action_loss, dist_entropy,
                  rl_setting, algorithm, _run):
    """
    Logs values to Observer and outputs the some numbers to command line.

    Args:
        j: Current gradient update
        num_updates: Total number of gradient updates to be performed
        T: total time passed
        id_tmp_dir: Working directory
        final_rewards: Total return on last completed episode
        tracking: `tracked_values` from function `track_values`
        num_ended_episodes: How many episodes have ended
        avg_nr_observer: Average number of non-blank observations in last batch and envs
        avg_encoding_loss: Encoding loss (i.e. L^{ELBO}) avg over last batch and envs
        total_loss: L = L^A + l^H*L^H + l^V*L^V + l^E*L^{ELBO}
        value_loss: L^V
        action_loss: L^A
        dist_entropy: L^H
            (all averaged over batch and environments)
        rl_setting: Config dict (see default.yaml for contents)
        algorithm: Config dict (see default.yaml for contents)
        _run: `Run` object from sacred. Needed to send stuff to the observer.
    """

    total_num_steps = (j + 1) * rl_setting['num_processes'] * rl_setting['num_steps']
    fps = int(total_num_steps / T)
    try:
        # The first few times the results might not be written to file yet
        true_results = load_results(id_tmp_dir)
        last_true_result = true_results.groupby('run_nr').last().mean().loc['r']
    except IndexError:
        last_true_result = 0

    num_frames = j * rl_setting['num_steps'] * rl_setting['num_processes']

    # Log scalars
    _run.log_scalar("result.true", last_true_result, num_frames)
    _run.log_scalar("result.mean", final_rewards.mean().item(), num_frames)
    _run.log_scalar("result.median", final_rewards.median().item(), num_frames)
    _run.log_scalar("result.min", final_rewards.min().item(), num_frames)
    _run.log_scalar("result.max", final_rewards.max().item(), num_frames)

    _run.log_scalar("particles.killed",
                    np.mean(tracking['num_killed_particles']),
                    num_frames)
    _run.log_scalar("episodes.num_ended", num_ended_episodes.item(), num_frames)
    _run.log_scalar("obs.fps", fps, num_frames)

    _run.log_scalar("obs.avg_nr_observed", avg_nr_observed)

    _run.log_scalar("loss.total", total_loss.item(), num_frames)
    _run.log_scalar("loss.value", value_loss.item(), num_frames)
    _run.log_scalar("loss.action", action_loss.item(), num_frames)
    _run.log_scalar("loss.entropy", dist_entropy.item(), num_frames)

    _run.log_scalar("loss.encoding", avg_encoding_loss.item(), num_frames)

    prior_loss_mean = emission_loss_mean = "-----"

    # When using the particle filter, save some additional statistics
    if algorithm['use_particle_filter']:
        stacked_prior_loss = torch.stack(tuple(tracking['prior_loss']))
        stacked_emission_loss = torch.stack(tuple(tracking['emission_loss']))
        prior_loss_mean = stacked_prior_loss.mean().item()
        emission_loss_mean = stacked_emission_loss.mean().item()

        _run.log_scalar("loss.prior.mean", prior_loss_mean, num_frames)
        _run.log_scalar("loss.emission.mean", emission_loss_mean, num_frames)

        if algorithm['particle_filter']['num_particles'] > 1:
            emission_loss_logsumexp = - math.logsumexp(
                - stacked_emission_loss,
                dim=2).mean()
            emission_loss_std = torch.std(
                stacked_emission_loss,
                dim=2).mean()
            prior_loss_logsumexp = - math.logsumexp(
                - stacked_prior_loss,
                dim=2).mean()
            prior_loss_std = torch.std(
                stacked_prior_loss,
                dim=2).mean()
            _run.log_scalar("loss.prior.logsumexp", prior_loss_logsumexp.item(), num_frames)
            _run.log_scalar("loss.prior.std", prior_loss_std.item(), num_frames)
            _run.log_scalar("loss.emission.logsumexp", emission_loss_logsumexp.item(), num_frames)
            _run.log_scalar("loss.emission.std", emission_loss_std.item(), num_frames)

    logging.info('Updt: {:5} |{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}||         |{:5}|{:5}|{:5}|{:5}||{:5}|{:5}|{:5}'.format(
        str(j / num_updates)[:5],
        str(fps),
        str(np.mean(tracking['num_killed_particles']))[:5],
        str(total_loss.item())[:5],
        str(final_rewards.mean().item())[:5],
        str(final_rewards.median().item())[:5],
        str(final_rewards.min().item())[:5],
        str(final_rewards.max().item())[:5],
        str(dist_entropy.item())[:5],
        str(value_loss.item())[:5],
        str(action_loss.item())[:5],
        str(avg_encoding_loss.item())[:5],
        str(prior_loss_mean)[:5],
        str(emission_loss_mean)[:5],
        str("-----")[:5]))


def get_environment_yaml(ex):
    """
    Get the name of the environment_yaml file that should be specified in the command line as:
    'python main.py -p with environment.config_file=<env_config_file>.yaml [...]'
    """
    _, _, usage = ex.get_usage()
    args = docopt(usage, [str(a) for a in sys.argv[1:]], help=False)
    config_updates, _ = get_config_updates(args['UPDATE'])
    # updates = arg_parser.get_config_updates(args['UPDATE'])[0]
    environment_yaml = config_updates.get('environment', {}).get('config_file', None)
    return environment_yaml
