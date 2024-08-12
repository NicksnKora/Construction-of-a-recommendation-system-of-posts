import psycopg2
from fastapi import FastAPI,HTTPException,Depends
import datetime
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from database import SessionLocal,engine
from table_feed import Post, User, Feed
from schema import UserGet, PostGet
from typing import List

app=FastAPI()

def get_db():
    with SessionLocal() as db:
        return db

@app.get("/post/recommendations", response_model=List[PostGet])
def get_recommendations(id: int, limit: int = 10, db: Session = Depends(get_db)):
    posts = db.query(Post.id,Post.text,Post.topic).select_from(Feed).filter(Feed.action=="like").join(Post, Post.id==Feed.post_id).group_by(Post.id,Post.text,Post.topic).order_by(func.count(Post.id).desc()).limit(limit).all()
    return posts

