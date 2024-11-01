import importlib


def load_sk_dataset(dataset: str)-> tuple:
    sk_module = importlib.import_module('sklearn.datasets')
    load_dataset = sk_module.__dict__[f'load_{dataset}']

    return load_dataset()
