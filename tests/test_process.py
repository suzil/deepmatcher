import io
import os
import shutil

import pytest
from conftest import embeddings, test_dir_path
from torchtext.utils import unicode_csv_reader

from deepmatcher import MatchingModel
from deepmatcher.data.process import (
    _check_header,
    _make_fields,
    process,
    process_unlabeled,
)


def test_check_header_1():
    path = os.path.join(test_dir_path, "test_datasets")
    a_dataset = "sample_table_small.csv"
    with io.open(
        os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8"
    ) as f:
        header = next(unicode_csv_reader(f))
    assert header == ["id", "left_a", "right_a", "label"]
    id_attr = "id"
    label_attr = "label"
    left_prefix = "left"
    right_prefix = "right"
    _check_header(header, id_attr, left_prefix, right_prefix, label_attr, [])


def test_check_header_2():
    path = os.path.join(test_dir_path, "test_datasets")
    a_dataset = "sample_table_small.csv"
    with io.open(
        os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8"
    ) as f:
        header = next(unicode_csv_reader(f))
    assert header == ["id", "left_a", "right_a", "label"]
    id_attr = "id"
    label_attr = "label"
    left_prefix = "left"
    right_prefix = "bb"
    with pytest.raises(ValueError):
        _check_header(header, id_attr, left_prefix, right_prefix, label_attr, [])


def test_check_header_3():
    path = os.path.join(test_dir_path, "test_datasets")
    a_dataset = "sample_table_small.csv"
    with io.open(
        os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8"
    ) as f:
        header = next(unicode_csv_reader(f))
    assert header == ["id", "left_a", "right_a", "label"]
    id_attr = "id"
    label_attr = "label"
    left_prefix = "aa"
    right_prefix = "right"
    with pytest.raises(ValueError):
        _check_header(header, id_attr, left_prefix, right_prefix, label_attr, [])


def test_check_header_5():
    path = os.path.join(test_dir_path, "test_datasets")
    a_dataset = "sample_table_small.csv"
    with io.open(
        os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8"
    ) as f:
        header = next(unicode_csv_reader(f))
    assert header == ["id", "left_a", "right_a", "label"]
    id_attr = "id"
    label_attr = ""
    left_prefix = "left"
    right_prefix = "right"
    with pytest.raises(ValueError):
        _check_header(header, id_attr, left_prefix, right_prefix, label_attr, [])


def test_check_header_6():
    path = os.path.join(test_dir_path, "test_datasets")
    a_dataset = "sample_table_small.csv"
    with io.open(
        os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8"
    ) as f:
        header = next(unicode_csv_reader(f))
    assert header == ["id", "left_a", "right_a", "label"]
    header.pop(1)
    id_attr = "id"
    label_attr = "label"
    left_prefix = "left"
    right_prefix = "right"
    with pytest.raises(AssertionError):
        _check_header(header, id_attr, left_prefix, right_prefix, label_attr, [])


def test_make_fields_1():
    path = os.path.join(test_dir_path, "test_datasets")
    a_dataset = "sample_table_large.csv"
    with io.open(
        os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8"
    ) as f:
        header = next(unicode_csv_reader(f))
    assert header == [
        "_id",
        "ltable_id",
        "rtable_id",
        "label",
        "ltable_Song_Name",
        "ltable_Artist_Name",
        "ltable_Price",
        "ltable_Released",
        "rtable_Song_Name",
        "rtable_Artist_Name",
        "rtable_Price",
        "rtable_Released",
    ]
    id_attr = "_id"
    label_attr = "label"
    fields = _make_fields(
        header, id_attr, label_attr, ["ltable_id", "rtable_id"], True, "nltk", True
    )
    assert len(fields) == 12
    counter = {}
    for tup in fields:
        if tup[1] not in counter:
            counter[tup[1]] = 0
        counter[tup[1]] += 1
    assert sorted(list(counter.values())) == [1, 1, 2, 8]


def test_process_1():
    vectors_cache_dir = ".cache"
    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)

    data_dir = os.path.join(test_dir_path, "test_datasets")
    train_path = "sample_table_large.csv"
    valid_path = "sample_table_large.csv"
    test_path = "sample_table_large.csv"
    cache_file = "cache.pth"
    cache_path = os.path.join(data_dir, cache_file)
    if os.path.exists(cache_path):
        os.remove(cache_path)

    process(
        data_dir,
        train=train_path,
        validation=valid_path,
        test=test_path,
        id_attr="_id",
        left_prefix="ltable_",
        right_prefix="rtable_",
        cache=cache_file,
        embeddings=embeddings,
        embeddings_cache_path="",
    )

    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)

    if os.path.exists(cache_path):
        os.remove(cache_path)


def test_process_unlabeled_1():
    vectors_cache_dir = ".cache"
    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)

    data_cache_path = os.path.join(test_dir_path, "test_datasets", "cacheddata.pth")
    if os.path.exists(data_cache_path):
        os.remove(data_cache_path)

    train, valid, test = process(
        path=os.path.join(test_dir_path, "test_datasets"),
        train="test_train.csv",
        validation="test_valid.csv",
        test="test_test.csv",
        id_attr="id",
        ignore_columns=("left_id", "right_id"),
        embeddings=embeddings,
        embeddings_cache_path="",
        pca=True,
    )

    model_save_path = "sif_model.pth"
    model = MatchingModel(attr_summarizer="sif")
    model.run_train(
        train,
        valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    test_unlabeled = process_unlabeled(
        path=os.path.join(test_dir_path, "test_datasets", "test_test.csv"),
        trained_model=model,
        ignore_columns=("left_id", "right_id"),
    )

    assert test_unlabeled.all_text_fields == test.all_text_fields

    if os.path.exists(model_save_path):
        os.remove(model_save_path)

    if os.path.exists(data_cache_path):
        os.remove(data_cache_path)

    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)
