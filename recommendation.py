# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import ml_metrics
from nptyping import NDArray
from typing import Dict, Any


def get_topk_artists_from_idxs(
    k: int,
    user_artists_df: pd.DataFrame(columns=["user_id", "artists"]),
    artists_idxs: NDArray[(Any,), int],
) -> NDArray[(Any,), int]:
    """
    Топ k самых часто встречаемых исполнителей из множества.
    По факту рекомендация самых популярных исполнителей в тернировочном множестве
    или кластере (в зависимости от artists_idxs)
    """
    train_artists_info = user_artists_df.loc[
        user_artists_df["artists"].isin(artists_idxs)
    ]
    top_artists = np.array(
        train_artists_info.groupby("artists")
        .count()
        .sort_values(by="user_id", ascending=False)
        .iloc[:k]
        .index
    )
    return top_artists


def get_similar_artists(
    artist_id: int,
    user_artists_df: pd.DataFrame(columns=["user_id", "artists"]),
    artists_train_array: NDArray[(Any,), int],
) -> NDArray[(Any,), int]:
    """
    Список исполнителей из тренировочного множества, которых слушают пользователи,
    слушающие конкретного исполнителя.
    """
    train_artists_info = user_artists_df.loc[
        user_artists_df["artists"].isin(artists_train_array)
    ]
    users = train_artists_info.loc[
        train_artists_info["artists"] == artist_id, "user_id"
    ]
    train_artists_info = user_artists_df.loc[
        user_artists_df["user_id"].isin(users), "artists"
    ]
    train_artists_info = train_artists_info.drop_duplicates()
    train_artists_info = train_artists_info.loc[
        train_artists_info != artist_id
    ]
    return train_artists_info.values


def get_cluster_recommendations_and_calculate_metrics(
    estimator,
    user_artists_df: pd.DataFrame(columns=["user_id", "artists"]),
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    train: NDArray[(Any,), int],
    test: NDArray[(Any,), int],
) -> Dict[int, float]:
    """
    Обучение модели кластеризации, поиск рекомендаций, оценка качества рекомендаций.
    """
    estimator = estimator.fit(train_matrix.toarray())
    train_df = pd.DataFrame([], columns=["artists", "clusters"])
    train_df["artists"] = train
    train_df["clusters"] = estimator.labels_
    train_df = train_df.groupby("clusters").agg(list)

    print("Основные статистики кластеризации:")
    print("Количество исполнителей: ", len(estimator.labels_))
    print("Количество кластеров:", len(np.unique(estimator.labels_)))
    counts_clusters = np.unique(estimator.labels_, return_counts=True)[1]
    print("Наибольший размер кластера:", np.max(counts_clusters))
    print("Гистограмма распределения количества кластеров по их величине")
    hist = np.histogram(
        counts_clusters, bins=[1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]
    )
    print("Деления: \n", hist[1])
    print("Частоты: \n", hist[0])

    test_df = pd.DataFrame([], columns=["artists", "clusters"])
    test_df["artists"] = test
    test_df["clusters"] = estimator.predict(test_matrix.toarray())
    test_df = test_df.groupby("clusters").agg(list).reset_index()

    metrics = {}
    for k in [1, 5, 10, 20]:
        metrics[k] = 0

    for cluster, test_artists in zip(test_df["clusters"], test_df["artists"]):
        # Формирование рекомендации как топ-21 исполнителей в соответствующем кластере
        if cluster in train_df.index:
            recommend = get_topk_artists_from_idxs(
                21, user_artists_df, train_df.loc[cluster]
            )
            for artist_id in test_artists:
                # Поиск списков исполнителей, которых слушают те же пользователи,
                # что и artist_id:
                users = np.unique(
                    user_artists_df.loc[
                        user_artists_df["artists"] == artist_id, "user_id"
                    ].values
                )
                similar_artists = (
                    user_artists_df.loc[
                        user_artists_df["user_id"].isin(users), "artists"
                    ]
                    .drop_duplicates()
                    .values
                )
                similar_artists = similar_artists[similar_artists != artist_id]

                # Подсчет метрик качества
                for key in metrics.keys():
                    metrics[key] += ml_metrics.apk(
                        actual=list(similar_artists),
                        predicted=list(recommend[:key]),
                        k=key,
                    )

    for key in metrics.keys():
        metrics[key] /= len(test)
        print("MAP@{}: ".format(key), metrics[key])
    return metrics
