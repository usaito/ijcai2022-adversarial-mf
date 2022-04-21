import argparse
import warnings
from pathlib import Path

import yaml

from utils.preprocessor import preprocess_yahoo_coat
from utils.result_tools import (compare_pscore_estimators,
                                identify_model_names, plot_damf_curves,
                                plot_rating_distributions, plot_test_curves,
                                save_experimental_results,
                                save_relative_results_of_damf,
                                summarize_data_statistics)

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", "-d", type=str, nargs="*", required=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    datasets = args.datasets
    num_sims = yaml.safe_load(open("../config.yaml", "rb"))["num_sims"]

    Path(f"../paper_results").mkdir(exist_ok=True, parents=True)
    summarize_data_statistics()

    for data in datasets:
        train, _, _, test, _, _ = preprocess_yahoo_coat(data=data)
        plot_path = Path(f"../paper_results/{data}/plots")
        plot_path.mkdir(exist_ok=True, parents=True)

        model_names = identify_model_names(data=data)
        save_experimental_results(data=data, model_names=model_names)
        save_relative_results_of_damf(data=data, model_names=model_names)
        compare_pscore_estimators(data=data)
        plot_rating_distributions(
            data=data, train=train, test=test, plot_path=plot_path
        )
        plot_test_curves(
            data=data, num_sims=num_sims, plot_path=plot_path, model_names=model_names
        )
        plot_damf_curves(data=data, num_sims=num_sims, plot_path=plot_path)
