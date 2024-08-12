import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserGet(BaseModel):
    age:int
    city:str
    country:str
    exp_group:int
    gender:int
    id:int
    os: str
    source:str

    class Config:
        orm_mode = True



class PostGet(BaseModel):
    id:int
    text:str
    topic:str

    class Config:
        orm_mode = True


class FeedGet(BaseModel):
    action:str
    post_id:int
    time:datetime.datetime
    user_id:int

    class Config:
        orm_mode = True