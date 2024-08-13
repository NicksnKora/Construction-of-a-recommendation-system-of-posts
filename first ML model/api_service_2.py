from fastapi import FastAPI, HTTPException
from typing import List
from datetime import datetime
import pandas as pd
from pydantic import BaseModel
import os
import pickle
from sqlalchemy import create_engine
from catboost import CatBoostClassifier


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model = CatBoostClassifier()
    model_path = get_model_path("C:/Users/User/PycharmProjects/Finale_project_2/catboost_model_v_2")
    model.load_model(model_path)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features_users() -> pd.DataFrame:
    query = 'SELECT * FROM nikita_koren_1998_user_id_lesson_22'
    return batch_load_sql(query)


def load_features_post() -> pd.DataFrame:
    query = 'SELECT * FROM nikita_koren_1998_post_feature_data_lesson_22'
    return batch_load_sql(query)


def table_post() -> pd.DataFrame:
    return batch_load_sql("SELECT * FROM public.post_text_df")


loaded_model = load_models()
df_user_data = load_features_users()
df_post_mod = load_features_post()
df_post_all = table_post()

features = ['total_interactions', 'like_count','topic','gender', 'country', 'city', 'exp_group','os','source', 'avg_age']

app = FastAPI()


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    # формируем таблицу с фичами под определенного user_id, обязательно сбрасываем индекс
    user_table = df_user_data[df_user_data['user_id'] == id].reset_index(drop=True)

    # добавляем timestamp для формирования фичей в таблицу user_table
    user_table['timestamp'] = time
    user_table['day_of_week'] = user_table['timestamp'].dt.dayofweek
    user_table['hour'] = user_table['timestamp'].dt.hour
    user_table = user_table.drop('timestamp', axis=1)

    # соединим user_table с df_post_text
    X = pd.merge(user_table, df_post_mod, how='cross', suffixes=('_user', '_post'))

    # получим список постов предсказательной модели
    post_id = pd.concat([X['post_id'], pd.DataFrame(loaded_model.predict_proba(X[features]).T[1],
                                                    columns=['prediction'])], axis=1).sort_values(by=['prediction'],
                                                                                                  ascending=False).head(
        limit)['post_id'].values

    # фильтруем датасет по полученным post_id
    result_table = df_post_all[df_post_all['post_id'].isin(post_id)].reset_index(drop=True)

    result = []
    for i in range(5):
        result.append(PostGet(id=result_table['post_id'].iloc[i],
                              text=result_table['text'].iloc[i],
                              topic=result_table['topic'].iloc[i]))

    if not result:
        raise HTTPException(404, "posts not found")
    return result
