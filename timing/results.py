import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_df(data, rename):
    for old, new in rename.items():
        data[new] = data[old]
    return data


def main():
    sns.set_style("whitegrid")
    data = pd.read_csv("./results.csv")
    sns.lineplot(
        prepare_df(
            data,
            {
                "num_nodes": "# nodes",
                "elapsed": "Time elapsed [seconds]",
                "model_name": "Model",
            },
        ),
        x="# nodes",
        y="Time elapsed [seconds]",
        hue="Model",
        marker="o",
    )
    plt.savefig("results.pdf")


if __name__ == "__main__":
    main()
