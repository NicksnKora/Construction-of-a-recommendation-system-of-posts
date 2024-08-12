from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer,String
from sqlalchemy.orm import relationship

from database import Base,engine, SessionLocal



class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text=Column(String)
    topic=Column(String)

class User(Base):
    __tablename__ = "user"
    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    id = Column(Integer,primary_key=True)
    os =Column(String)
    source= Column(String)

class Feed(Base):
    __tablename__ = "feed_action"
    id=Column(Integer,primary_key=True)
    post_id=Column(Integer, ForeignKey("post.id", primary_key=True))
    post = relationship("Post")
    user_id=Column(Integer, ForeignKey("user.id", primary_key=True))
    user = relationship("User")
    action=Column(String)
    time=Column(TIMESTAMP)


if __name__ == '__main__':
    Base.metadata.create_all()


