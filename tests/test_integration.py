import os
import shutil
from collections import namedtuple

import pytest
from conftest import embeddings, test_dir_path

from deepmatcher import MatchingModel, attr_summarizers
from deepmatcher.data.process import process, process_unlabeled

Datasets = namedtuple("Datasets", "train valid test")


@pytest.fixture(scope="module")
def datasets():
    vectors_cache_dir = ".cache"
    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)

    data_cache_path = os.path.join(test_dir_path, "test_datasets", "train_cache.pth")
    if os.path.exists(data_cache_path):
        os.remove(data_cache_path)

    train, valid, test = process(
        path=os.path.join(test_dir_path, "test_datasets"),
        cache="train_cache.pth",
        train="test_train.csv",
        validation="test_valid.csv",
        test="test_test.csv",
        embeddings=embeddings,
        embeddings_cache_path="",
        ignore_columns=("left_id", "right_id"),
    )
    yield Datasets(train, valid, test)

    if os.path.exists(data_cache_path):
        os.remove(data_cache_path)

    if os.path.exists(vectors_cache_dir):
        shutil.rmtree(vectors_cache_dir)


def test_train_save_load_sif(datasets):
    model_save_path = "sif_model.pth"
    model = MatchingModel(attr_summarizer="sif")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )
    s1 = model.run_eval(datasets.test)

    model2 = MatchingModel(attr_summarizer="sif")
    model2.load_state(model_save_path)
    s2 = model2.run_eval(datasets.test)

    assert s1 == s2

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_train_save_load_rnn(datasets):
    model_save_path = "rnn_model.pth"
    model = MatchingModel(attr_summarizer="rnn")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )
    s1 = model.run_eval(datasets.test)

    model2 = MatchingModel(attr_summarizer="rnn")
    model2.load_state(model_save_path)
    s2 = model2.run_eval(datasets.test)

    assert s1 == s2

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_train_save_load_attention(datasets):
    model_save_path = "attention_model.pth"
    model = MatchingModel(attr_summarizer="attention")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    s1 = model.run_eval(datasets.test)

    model2 = MatchingModel(attr_summarizer="attention")
    model2.load_state(model_save_path)
    s2 = model2.run_eval(datasets.test)

    assert s1 == s2

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_train_save_load_hybrid(datasets):
    model_save_path = "hybrid_model.pth"
    model = MatchingModel(attr_summarizer="hybrid")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    s1 = model.run_eval(datasets.test)

    model2 = MatchingModel(attr_summarizer="hybrid")
    model2.load_state(model_save_path)
    s2 = model2.run_eval(datasets.test)

    assert s1 == s2

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_train_save_load_hybrid_self_attention(datasets):
    model_save_path = "self_att_hybrid_model.pth"
    model = MatchingModel(
        attr_summarizer=attr_summarizers.Hybrid(word_contextualizer="self-attention")
    )

    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    s1 = model.run_eval(datasets.test)

    model2 = MatchingModel(
        attr_summarizer=attr_summarizers.Hybrid(word_contextualizer="self-attention")
    )
    model2.load_state(model_save_path)
    s2 = model2.run_eval(datasets.test)

    assert s1 == s2

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_predict_unlabeled_sif(datasets):
    model_save_path = "sif_model.pth"
    model = MatchingModel(attr_summarizer="sif")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    unlabeled = process_unlabeled(
        path=os.path.join(test_dir_path, "test_datasets", "test_unlabeled.csv"),
        trained_model=model,
        ignore_columns=("left_id", "right_id"),
    )

    pred_test = model.run_eval(datasets.test, return_predictions=True)
    pred_unlabeled = model.run_prediction(unlabeled)

    assert sorted(tup[1] for tup in pred_test) == sorted(
        list(pred_unlabeled["match_score"])
    )

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_predict_unlabeled_rnn(datasets):
    model_save_path = "rnn_model.pth"
    model = MatchingModel(attr_summarizer="rnn")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    unlabeled = process_unlabeled(
        path=os.path.join(test_dir_path, "test_datasets", "test_test.csv"),
        trained_model=model,
        ignore_columns=("left_id", "right_id", "label"),
    )

    pred_test = model.run_eval(datasets.test, return_predictions=True)
    pred_unlabeled = model.run_prediction(unlabeled)

    assert sorted(tup[1] for tup in pred_test) == sorted(
        list(pred_unlabeled["match_score"])
    )

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_predict_unlabeled_attention(datasets):
    model_save_path = "attention_model.pth"
    model = MatchingModel(attr_summarizer="attention")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    unlabeled = process_unlabeled(
        path=os.path.join(test_dir_path, "test_datasets", "test_unlabeled.csv"),
        trained_model=model,
        ignore_columns=("left_id", "right_id"),
    )

    pred_test = model.run_eval(datasets.test, return_predictions=True)
    pred_unlabeled = model.run_prediction(unlabeled)

    assert sorted(tup[1] for tup in pred_test) == sorted(
        list(pred_unlabeled["match_score"])
    )

    if os.path.exists(model_save_path):
        os.remove(model_save_path)


def test_predict_unlabeled_hybrid(datasets):
    model_save_path = "hybrid_model.pth"
    model = MatchingModel(attr_summarizer="hybrid")
    model.run_train(
        datasets.train,
        datasets.valid,
        epochs=1,
        batch_size=8,
        best_save_path=model_save_path,
        pos_neg_ratio=3,
    )

    unlabeled = process_unlabeled(
        path=os.path.join(test_dir_path, "test_datasets", "test_unlabeled.csv"),
        trained_model=model,
        ignore_columns=("left_id", "right_id"),
    )

    pred_test = model.run_eval(datasets.test, return_predictions=True)
    pred_unlabeled = model.run_prediction(unlabeled)

    assert sorted(tup[1] for tup in pred_test) == sorted(
        list(pred_unlabeled["match_score"])
    )

    if os.path.exists(model_save_path):
        os.remove(model_save_path)
