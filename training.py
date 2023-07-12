import hydra
import pandas
import argparse
from src.demo import train
from matplotlib import lines
import matplotlib.pyplot as plt
from omegaconf import DictConfig

parser = argparse.ArgumentParser()
parser.add_argument('--config-name', type=str)
args = parser.parse_args()
config_name = args.config_name

def plot(data):
    line_types = list(lines.lineStyles.keys())
    line_markers = list(lines.lineMarkers.keys())
    colors = {'rewards': 'tab:green', 'safety': 'tab:blue'}

    fig, axs = plt.subplots(1, len(data), figsize=(40, 10))
    legends = set()
    i = 0
    for metric in data.keys():
        ax = axs[i]
        Y = data[metric].rolling(window=75, min_periods=1).mean()

        X = list(range(Y.shape[0]))
        l, = ax.plot(X, Y, line_types[2], color=colors[metric], linewidth=1.5, markersize=2, \
                marker=line_markers[4])
        legends.add(l)

        ax.grid(True, which='major', linestyle=':')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if metric == 'rewards':
            ax.set_ylabel(metric.upper() + '.ROLLING', fontdict={'fontsize': 30})

        else:
            ax.set_ylabel(metric.upper() + '.RATE', fontdict={'fontsize': 30})
        i += 1

    print("***")
    ax.xaxis.set_tick_params(labelsize='large')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)


    plt.legend(loc='upper center', bbox_to_anchor=(-1.3, -0.15), ncol=6, fancybox=True, shadow=True, fontsize=30)
    plt.show()


@hydra.main(config_path="conf", config_name=config_name)
def training(cfg: DictConfig):
    model = train(cfg)
    rewards = pandas.DataFrame(model.rewards)
    safety_rates = pandas.DataFrame(model.violates)

    rolling_rewards = rewards.rolling(window=100, min_periods=1).mean()
    accumulated_rates = safety_rates.rolling(window=10000, min_periods=1).mean()
    data = {"rewards": rolling_rewards, "safety": accumulated_rates}
    print("The resulting accumulated safety rate of this run is {}, with a rolling reward of{}".format(
        accumulated_rates.iloc[-1].tolist(), rolling_rewards.iloc[-1].tolist()))
    plot(data)

if __name__ == "__main__":
    model = training()