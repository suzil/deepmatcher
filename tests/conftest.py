import os

import pytest
from torchtext.vocab import FastText

test_dir_path = os.path.dirname(os.path.realpath(__file__))

embeddings = FastText(language="hsb", max_vectors=50)


@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(test_dir_path, "test_datasets")


@pytest.fixture(scope="module")
def train_filename():
    return "test_train.csv"


@pytest.fixture(scope="module")
def valid_filename():
    return "test_valid.csv"


@pytest.fixture(scope="module")
def test_filename():
    return "test_test.csv"
