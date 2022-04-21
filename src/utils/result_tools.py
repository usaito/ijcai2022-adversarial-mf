"""Some tools for summarizing experimental results."""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly.graph_objs import Figure, Histogram, Layout, Scatter
from plotly.offline import plot

from utils.preprocessor import preprocess_yahoo_coat

metrics = ["mse", "ndcg", "recall"]
datasets = ["yahoo", "coat"]
stats_idx = [
    "#User",
    "#Item",
    "#Rating",
    "Sparsity",
    "Avg. rating of train",
    "Avg. rating of test",
    "KL divergence",
]
colors = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
    "rgb(148, 103, 189)",
    "rgb(140, 86, 75)",
    "rgb(227, 119, 194)",
    "rgb(127, 127, 127)",
    "rgb(188, 189, 34)",
    "rgb(23, 190, 207)",
]
fill_colors = [
    "rgba(31, 119, 180, 0.2)",
    "rgba(255, 127, 14, 0.2)",
    "rgba(44, 160, 44, 0.2)",
    "rgba(214, 39, 40, 0.2)",
    "rgba(148, 103, 189, 0.2)",
    "rgba(140, 86, 75, 0.2)",
    "rgba(227, 119, 194, 0.2)",
    "rgba(127, 127, 127, 0.2)",
    "rgba(188, 189, 34, 0.2)",
    "rgba(23, 190, 207, 0.2)",
]


def kl_div(train: np.ndarray, test: np.ndarray) -> float:
    """Estimate KL divergence of rating distributions
    between training and test sets."""
    p = (
        np.unique(train[:, 2], return_counts=True)[1]
        / np.unique(train[:, 2], return_counts=True)[1].sum()
    )
    q = (
        np.unique(test[:, 2], return_counts=True)[1]
        / np.unique(test[:, 2], return_counts=True)[1].sum()
    )
    return np.round(np.sum(np.where(p != 0, p * np.log(p / q), 0)), 4)


def summarize_data_statistics() -> None:
    """Save dataset statistics with Tex Table Format."""
    stat_data_list = []
    for data in datasets:
        train, _, _, test, num_users, num_items = preprocess_yahoo_coat(data=data)

        num_data = train.shape[0]
        spasity = f"{100 * (num_data / (num_users * num_items)).round(4)}%"
        avg_train, avg_test = train[:, 2].mean().round(3), test[:, 2].mean().round(3)
        kl = kl_div(train, test)
        stat_data = DataFrame(
            data=[num_users, num_items, num_data, spasity, avg_train, avg_test, kl],
            index=stats_idx,
            columns=[data],
        ).T
        stat_data_list.append(stat_data)
    pd.concat(stat_data_list).to_csv("../paper_results/data_stat.csv", sep="&")


def save_experimental_results(data: str, model_names: List[str]) -> None:
    """Save results with Tex Table format."""
    raw_results_path = Path(f"../logs/{data}")
    paper_results_path = Path(f"../paper_results/{data}")
    paper_results_path.mkdir(exist_ok=True, parents=True)

    results_mse_dict = {}
    results_ndcg_dict = {}
    results_recall_dict = {}
    for model in model_names:
        results_ = pd.read_csv(
            str(raw_results_path / f"{model}/results.csv"), index_col=0
        )
        results_mse_dict[model] = results_["MSE"]
        results_ndcg_dict[model] = results_["nDCG"]
        results_recall_dict[model] = results_["Recall"]

    results_mse = DataFrame(results_mse_dict).describe().round(4).T
    results_ndcg = DataFrame(results_ndcg_dict).describe().round(4).T
    results_recall = DataFrame(results_recall_dict).describe().round(4).T
    results_list = [results_mse, results_ndcg, results_recall]
    results_dict = {}
    for results, metric in zip(results_list, metrics):
        results_dict[metric.upper()] = results["mean"].apply(str) + results[
            "std"
        ].apply(lambda x: f" ($pm$ {str(x)}) ")
    DataFrame(results_dict).to_csv(
        str(paper_results_path / "overall_results.csv"), sep="&"
    )


def save_relative_results_of_damf(data: str, model_names: List[str]) -> None:
    """Save results with Tex Table format."""
    raw_results_path = Path(f"../logs/{data}")
    paper_results_path = Path(f"../paper_results/{data}")
    paper_results_path.mkdir(exist_ok=True, parents=True)

    rel_dict = {m: dict() for m in model_names}
    for model in model_names:
        results_damf = pd.read_csv(raw_results_path / f"damf/results.csv", index_col=0)
        for metric in ["MSE", "nDCG", "Recall"]:
            results_ = pd.read_csv(
                raw_results_path / f"{model}/results.csv", index_col=0
            )
            rel = np.mean(results_damf[metric].values / results_[metric].values)
            rel_dict[model][metric] = (100 * (rel - 1)).round(2)

    pd.DataFrame(rel_dict).T.to_csv(str(paper_results_path / "relative_results.csv"))


def compare_pscore_estimators(data: str) -> None:
    """Save results with Tex Table format."""
    raw_results_path = Path(f"../logs/{data}")
    paper_results_path = Path(f"../paper_results/{data}")
    paper_results_path.mkdir(exist_ok=True, parents=True)

    models = ["ips", "dr"]
    pscores = ["user", "item", "both", "1bitmc"]
    rel_mse_dict = {m: dict() for m in models}
    for model in models:
        results_oracle = pd.read_csv(
            raw_results_path / f"{model}-oracle/results.csv", index_col=0
        )
        for pscore in pscores:
            results_ = pd.read_csv(
                raw_results_path / f"{model}-{pscore}/results.csv", index_col=0
            )
            mse = np.mean(results_["MSE"].values)
            rel_mse = 100 * (
                np.mean(results_["MSE"].values / results_oracle["MSE"].values) - 1
            )
            rel_mse_dict[model][pscore] = f"{mse.round(4)} (+{(rel_mse).round(1)}%) "

    pd.DataFrame(rel_mse_dict).T[pscores].to_csv(
        paper_results_path / "rel_mse.csv", sep="&"
    )


def plot_rating_distributions(
    data: str, train: np.ndarray, test: np.ndarray, plot_path: Path
) -> None:
    """Plot rating distribution bar plots."""
    hist_train = Histogram(x=train[:, 2], name="Train", histnorm="probability")
    hist_test = Histogram(x=test[:, 2], name="Test", histnorm="probability")

    plot(
        Figure(data=[hist_train, hist_test], layout=layout_dist),
        filename=str(plot_path / f"rate_dist.html"),
        auto_open=False,
    )


def transform_model_name(model_name: str) -> str:
    """Transform raw model name into a name used in the plots."""
    if "oracle" in model_name:
        if "ips" in model_name:
            return "MF-IPS (true)"
        else:
            return "MF-DR (true)"
    elif "ips" in model_name:
        if "uniform" in model_name:
            return "MF"
        else:
            return "MF-IPS"
    elif "dr" in model_name:
        return "MF-DR"
    elif model_name == "damf":
        return "DAMF"
    elif model_name == "cause":
        return "CausE"


def identify_model_names(data: str) -> List[str]:
    """Identify model names that are to be reported in Tables and Figures in the paper."""
    raw_results_path = Path(f"../logs/{data}")
    model_names = []
    pscores = ["item", "user", "both", "1bitmc"]

    for est in ["ips", "dr"]:
        best_mse, best_model = 1e5, ""
        for pscore in pscores:
            model_ = est + "-" + pscore
            mse_ = pd.read_csv(
                str(raw_results_path / f"{model_}/results.csv"), index_col=0
            )
            mse = mse_.describe().loc["mean", "MSE"]
            if mse < best_mse:
                best_mse, best_model = mse, model_
        model_names.append(best_model)

    return ["ips-uniform"] + model_names + ["damf", "cause", "ips-oracle", "dr-oracle"]


def plot_test_curves(
    data: str, num_sims: int, plot_path: Path, model_names: List[str]
) -> None:
    """Plot loss curves of methods on test data."""
    scatter_list = []
    for i, model in enumerate(model_names):
        loss_path = Path(f"../logs/{data}/{model}/loss")
        test = np.concatenate(
            [
                np.expand_dims(np.load(str(loss_path / f"test_{i}.npy")), 1)
                for i in np.arange(num_sims)
            ],
            1,
        )
        upper, lower = test.mean(1) + test.std(1), test.mean(1) - test.std(1)
        scatter_list.append(
            Scatter(
                x=np.arange(len(test)),
                y=test.mean(1),
                name=transform_model_name(model),
                opacity=0.8,
                mode="lines",
                line=dict(color=colors[i], width=5),
            )
        )
        scatter_list.append(
            Scatter(
                x=np.r_[np.arange(len(test)), np.arange(len(test))[::-1]],
                y=np.r_[upper, lower[::-1]],
                showlegend=False,
                fill="tozerox",
                fillcolor=fill_colors[i],
                mode="lines",
                line=dict(color="rgba(255,255,255,0)"),
            )
        )

    plot(
        Figure(data=scatter_list, layout=layout_test_curves),
        filename=str(plot_path / f"test_curves.html"),
        auto_open=False,
    )


def plot_damf_curves(data: str, num_sims: int, plot_path: Path) -> None:
    """Plot loss curves of methods on test data."""
    scatter_list = []
    loss_path = Path(f"../logs/{data}/damf/loss")
    val = np.concatenate(
        [
            np.expand_dims(np.load(str(loss_path / f"val_{i}.npy")), 1)
            for i in np.arange(num_sims)
        ],
        1,
    )
    test = np.concatenate(
        [
            np.expand_dims(np.load(str(loss_path / f"test_{i}.npy")), 1)
            for i in np.arange(num_sims)
        ],
        1,
    )
    for i, (loss, name) in enumerate(zip([val, test], ["Upper Bound", "Ideal Loss"])):
        upper, lower = loss.mean(1) + loss.std(1), loss.mean(1) - loss.std(1)
        scatter_list.append(
            Scatter(
                x=np.arange(len(loss)),
                y=loss.mean(1),
                name=name,
                opacity=0.8,
                mode="lines",
                line=dict(color=colors[i], width=10),
            )
        )
        scatter_list.append(
            Scatter(
                x=np.r_[np.arange(len(loss)), np.arange(len(loss))[::-1]],
                y=np.r_[upper, lower[::-1]],
                showlegend=False,
                fill="tozerox",
                fillcolor=fill_colors[i],
                mode="lines",
                line=dict(color="rgba(255,255,255,0)"),
            )
        )

    plot(
        Figure(data=scatter_list, layout=layout_damf_curves),
        filename=str(plot_path / f"damf_curves.html"),
        auto_open=False,
    )


layout_dist = Layout(
    paper_bgcolor="rgb(255,255,255)",
    plot_bgcolor="rgb(232,232,232)",
    width=1600,
    height=950,
    xaxis=dict(
        title="Rating Values",
        titlefont=dict(size=45),
        tickfont=dict(size=35),
        gridcolor="rgb(255,255,255)",
    ),
    yaxis=dict(
        title="Probability Mass",
        titlefont=dict(size=45),
        tickfont=dict(size=25),
        gridcolor="rgb(255,255,255)",
    ),
    legend=dict(
        bgcolor="rgb(245,245,245)",
        x=0.98,
        xanchor="right",
        y=0.99,
        yanchor="top",
        font=dict(size=40),
    ),
    margin=dict(l=120, t=30, b=115),
)

layout_test_curves = Layout(
    paper_bgcolor="rgb(255,255,255)",
    plot_bgcolor="rgb(235,235,235)",
    width=1600,
    height=950,
    xaxis=dict(
        title="Iterations",
        titlefont=dict(size=45),
        tickfont=dict(size=35),
        gridcolor="rgb(255,255,255)",
    ),
    yaxis=dict(
        title="Mean Squared Error (±StdDev)",
        range=[0.8, 2.5],
        titlefont=dict(size=40),
        tickfont=dict(size=28),
        gridcolor="rgb(255,255,255)",
    ),
    legend=dict(
        bgcolor="rgb(245,245,245)",
        x=0.5,
        orientation="h",
        xanchor="center",
        y=1.03,
        yanchor="bottom",
        font=dict(size=35),
    ),
    margin=dict(l=110, t=50, b=115),
)


layout_damf_curves = Layout(
    paper_bgcolor="rgb(255,255,255)",
    plot_bgcolor="rgb(232,232,232)",
    width=1600,
    height=950,
    xaxis=dict(
        title="Iterations",
        titlefont=dict(size=45),
        tickfont=dict(size=35),
        gridcolor="rgb(255,255,255)",
    ),
    yaxis=dict(
        title="Mean Squared Error (±StdDev)",
        range=[0.8, 2.0],
        titlefont=dict(size=40),
        tickfont=dict(size=28),
        gridcolor="rgb(255,255,255)",
    ),
    legend=dict(
        bgcolor="rgb(245,245,245)",
        x=0.98,
        xanchor="right",
        y=0.99,
        yanchor="top",
        orientation="h",
        font=dict(size=45),
    ),
    margin=dict(l=120, t=30, b=115),
)
