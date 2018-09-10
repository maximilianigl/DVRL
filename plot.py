import matplotlib.pyplot as plt
import json

import argparse
parser = argparse.ArgumentParser(description='Very simple plotting script.')
parser.add_argument('--id', action="store", dest="id")
parser.add_argument('--metric', action="store", dest="metric", default='result.true')

# Should be grouped by label x condition1 x condition2
def plot_metric(filename, metric_name):
    with open(filename) as f:
        data = json.load(f)

    x = data[metric_name]['steps']
    y = data[metric_name]['values']
    plt.plot(x, y)
    plt.xlabel("Frames")
    plt.ylabel(metric_name)
    plt.savefig("Result.png")


if __name__ == "__main__":
    args = parser.parse_args()
    plot_metric("./saved_runs/" + str(args.id) +"/metrics.json", args.metric)
