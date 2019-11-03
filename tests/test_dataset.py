import io
import os
import shutil

import pandas as pd
import pytest
from torchtext.utils import unicode_csv_reader

from deepmatcher.data.dataset import MatchingDataset, split
from deepmatcher.data.field import MatchingField
from deepmatcher.data.process import _make_fields, process
from tests import embeddings, test_dir_path


def test_class_matching_dataset():
    fields = [("left_a", MatchingField()), ("right_a", MatchingField())]
    col_naming = {"id": "id", "label": "label", "left": "left", "right": "right"}
    path = os.path.join(test_dir_path, "test_datasets", "sample_table_small.csv")
    md = MatchingDataset(fields, col_naming, path=path)
    assert md.id_field == "id"
    assert md.label_field == "label"
    assert md.all_left_fields == ["left_a"]
    assert md.all_right_fields == ["right_a"]
    assert md.all_text_fields == ["left_a", "right_a"]
    assert md.canonical_text_fields == ["_a"]


@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(test_dir_path, "test_datasets")


@pytest.fixture(scope="module")
def train():
    return "test_train.csv"


@pytest.fixture(scope="module")
def validation():
    return "test_valid.csv"


@pytest.fixture(scope="module")
def test():
    return "test_test.csv"


@pytest.fixture(scope="module")
def cache_name():
    return "test_cacheddata.pth"


@pytest.fixture(scope="module")
def id_attr():
    return "id"


@pytest.fixture(scope="module")
def label_attr():
    return "label"


@pytest.fixture(scope="module")
def fields(id_attr, label_attr, data_dir, train):
    ignore_columns = ["left_id", "right_id"]
    with io.open(
        os.path.expanduser(os.path.join(data_dir, train)), encoding="utf8"
    ) as f:
        header = next(unicode_csv_reader(f))
    return _make_fields(
        header, id_attr, label_attr, ignore_columns, True, "nltk", False
    )


@pytest.fixture(scope="module")
def column_naming(id_attr, label_attr):
    return {"id": id_attr, "left": "left_", "right": "right_", "label": label_attr}


@pytest.fixture
def remove_cache(data_dir, cache_name):
    yield
    cache_name = os.path.join(data_dir, cache_name)
    if os.path.exists(cache_name):
        os.remove(cache_name)


def test_splits_1(
    data_dir, train, validation, test, fields, column_naming, cache_name, remove_cache
):
    datasets = MatchingDataset.splits(
        data_dir,
        train,
        validation,
        test,
        fields,
        None,
        None,
        column_naming,
        cache_name,
        train_pca=False,
    )
    assert datasets


def test_splits_2(
    data_dir, train, validation, test, fields, column_naming, cache_name, remove_cache
):
    datasets = MatchingDataset.splits(
        data_dir,
        train,
        validation,
        test,
        fields,
        None,
        None,
        column_naming,
        cache_name,
        train_pca=False,
    )
    assert datasets

    with pytest.raises(MatchingDataset.CacheStaleException):
        MatchingDataset.splits(
            data_dir,
            "sample_table_small.csv",
            validation,
            test,
            fields,
            None,
            None,
            column_naming,
            cache_name,
            True,
            False,
            train_pca=False,
        )


def test_splits_3(
    data_dir, train, validation, test, fields, column_naming, cache_name, remove_cache
):
    datasets = MatchingDataset.splits(
        data_dir,
        train,
        validation,
        test,
        fields,
        None,
        None,
        column_naming,
        cache_name,
        train_pca=False,
    )
    assert datasets

    datasets_2 = MatchingDataset.splits(
        data_dir,
        train,
        validation,
        test,
        fields,
        None,
        None,
        column_naming,
        cache_name,
        False,
        False,
        train_pca=False,
    )
    assert datasets_2


def test_split_1():
    labeled_path = os.path.join(
        test_dir_path, "test_datasets", "sample_table_large.csv"
    )
    labeled_table = pd.read_csv(labeled_path)
    ori_cols = list(labeled_table.columns)
    out_path = os.path.join(test_dir_path, "test_datasets")
    train_prefix = "train.csv"
    valid_prefix = "valid.csv"
    test_prefix = "test.csv"
    split(labeled_table, out_path, train_prefix, valid_prefix, test_prefix)

    train_path = os.path.join(out_path, train_prefix)
    valid_path = os.path.join(out_path, valid_prefix)
    test_path = os.path.join(out_path, test_prefix)

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)

    assert list(train.columns) == ori_cols
    assert list(valid.columns) == ori_cols
    assert list(test.columns) == ori_cols

    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(valid_path):
        os.remove(valid_path)
    if os.path.exists(test_path):
        os.remove(test_path)


def test_split_2():
    labeled_path = os.path.join(
        test_dir_path, "test_datasets", "sample_table_large.csv"
    )
    labeled_table = pd.read_csv(labeled_path)
    ori_cols = list(labeled_table.columns)
    out_path = os.path.join(test_dir_path, "test_datasets")
    train_prefix = "train.csv"
    valid_prefix = "valid.csv"
    test_prefix = "test.csv"
    split(labeled_path, out_path, train_prefix, valid_prefix, test_prefix)

    train_path = os.path.join(out_path, train_prefix)
    valid_path = os.path.join(out_path, valid_prefix)
    test_path = os.path.join(out_path, test_prefix)

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)

    assert list(train.columns) == ori_cols
    assert list(valid.columns) == ori_cols
    assert list(test.columns) == ori_cols

    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(valid_path):
        os.remove(valid_path)
    if os.path.exists(test_path):
        os.remove(test_path)


def test_get_raw_table():
    vectors_cache_dir = ".cache"
    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)

    data_cache_path = os.path.join(test_dir_path, "test_datasets", "cacheddata.pth")
    if os.path.exists(data_cache_path):
        os.remove(data_cache_path)

    train = process(
        path=os.path.join(test_dir_path, "test_datasets"),
        train="sample_table_small.csv",
        id_attr="id",
        embeddings=embeddings,
        embeddings_cache_path="",
        pca=False,
    )

    train_raw = train.get_raw_table()
    ori_train = pd.read_csv(
        os.path.join(test_dir_path, "test_datasets", "sample_table_small.csv")
    )
    assert set(train_raw.columns) == set(ori_train.columns)

    if os.path.exists(data_cache_path):
        os.remove(data_cache_path)

    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)
