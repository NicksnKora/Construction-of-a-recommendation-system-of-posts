import os

from datetime import datetime
from typing import List

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import create_engine


app = FastAPI()
model_filename = 'catboost_model_june.cbm'
engine = create_engine(
    url="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml",
    pool_size=20, max_overflow=0)

featured_posts_base_name = 'n_koren_3_posts_featured_df_june'


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        from_attributes = True
        validate_default = True


def batch_load_sql(query: str) -> pd.DataFrame:
    chunksize = 200000

    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=chunksize):
        chunks.append(chunk_dataframe)
    conn.close()
    logger.info(f'{len(chunk_dataframe)} strings loaded')
    return pd.concat(chunks, ignore_index=True)


def get_model_path(filename: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f'/workdir/user_input/model'
    else:
        MODEL_PATH = f'./model/{filename}'
    return MODEL_PATH


def load_models():
    model_path = get_model_path(model_filename)
    model = CatBoostClassifier().load_model(model_path)
    logger.info('Models loaded')
    return model


logger.info('Featured posts loading...')
featured_posts_df = batch_load_sql(f'SELECT * FROM {featured_posts_base_name}')
logger.info('Users data loading...')
users_data = batch_load_sql(f'SELECT * FROM user_data')
logger.info('Users likes loading...')
liked_posts = batch_load_sql("SELECT DISTINCT user_id, post_id FROM feed_data WHERE action='like'")
# liked_posts = pd.read_csv('liked_posts.csv')
posts = featured_posts_df[['post_id', 'text', 'topic']].set_index('post_id')
logger.info('Posts is loaded')

model = load_models()


def get_recommended_feed(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    request_start = datetime.now()

    user_features = users_data.loc[users_data.user_id == id].drop(columns='user_id')
    user_features['hour'] = time.hour
    user_features['month'] = time.month
    user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info(f'User features created, time_delta = {datetime.now() - request_start}')

    featured_posts = featured_posts_df
    posts_user_features = featured_posts.assign(**user_features)
    logger.info(f'User data assigned, time_delta = {datetime.now() - request_start}')

    cols_order = [
        'topic', 'TextCluster', 'dist_to_cluster_1', 'dist_to_cluster_2',
        'dist_to_cluster_3', 'dist_to_cluster_4', 'dist_to_cluster_5',
        'dist_to_cluster_6', 'dist_to_cluster_7', 'dist_to_cluster_8',
        'dist_to_cluster_9', 'dist_to_cluster_10', 'dist_to_cluster_11',
        'dist_to_cluster_12', 'dist_to_cluster_13', 'dist_to_cluster_14',
        'dist_to_cluster_15', 'age', 'city', 'country',
        'exp_group', 'gender', 'os', 'source', 'hour', 'month'
    ]

    object_cols = ['topic', 'TextCluster', 'age', 'city', 'country',
                   'exp_group', 'gender', 'os', 'source', 'hour', 'month']

    posts_user_features = posts_user_features[cols_order]
    logger.info(f'Posts_user table created, time_delta = {datetime.now() - request_start}')

    predictions = model.predict_proba(posts_user_features)[:, 1]
    posts['prediction'] = predictions
    logger.info(f'Predictions created, time_delta = {datetime.now() - request_start}')

    already_liked_ids = liked_posts[liked_posts.user_id == id].post_id.values
    recommended_idx = (posts[~posts.index.isin(already_liked_ids)]
                       .sort_values('prediction', ascending=False)
                       .head(limit)).index

    logger.info(f'Responding recommended posts, time_delta = {datetime.now() - request_start}')
    return [
        PostGet(**{
            "id": i,
            "text": posts.loc[i, 'text'],
            "topic": posts.loc[i, 'topic']
        }) for i in recommended_idx
    ]


@app.get("/post/recommendations", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)
