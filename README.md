## Towards Resolving Propensity Contradiction in Offline Recommender Learning

### About

This repository contains the code to replicate the experiments conducted in the paper "Towards Resolving Propensity Contradiction in Offline Recommender Learning" accepted at [IJCAI2022](https://ijcai-22.org/).

If you find this code useful in your research then please site:
```
@inproceedings{saito2022towards,
  author = {Saito, Yuta and Nomura, Masahiro},
  title = {Towards Resolving Propensity Contradiction in Offline Recommender Learning},
  booktitle = {Proceedings of the 31st International Joint Conference on Artificial Intelligence},
  pages = {xxx-xxx},
  year = {2022},
}
```

### Dependencies

- numpy==1.19.1
- pandas==1.1.2
- optuna==0.17.0
- scikit-learn==0.23.1
- tensorflow==1.15.4
- plotly==3.10.0
- pyyaml==5.1.2

### Datasets

To run the experiments with real-world datasets, the following datasets need to be prepared.

1. download the [Coat dataset](https://www.cs.cornell.edu/~schnabts/mnar/) and put `train.ascii` and `test.ascii` files into `./data/coat/` directory.
2. download the [Yahoo! R3 dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r) and put `train.txt` and `test.txt` files into `./data/yahoo/` directory.

It should be noted that we use the original Yahoo! R3 and Coat datasets, as they contain MCAR test data.

### Running the code

Navigate to the `src` directory and run the command

```bash
for data in yahoo coat
do
  python main.py -d $data -m damf -t
done
```

This will run experiments conducted in Section 4 in the paper. You can see the default settings used in our experiments in the [`config.yaml`](./config.yaml) file.

Once the simulations of all methods are finished executing, you can summarize the results reported in Table 1 by running the command below in `./src/` directory.

```zsh
python summarize_results.py -d $data yahoo coat
```

Then, you can find the results in `./paper_results/` directory.
