from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Game(Base):
    __tablename__ = 'games'

    game_id = Column(Integer, primary_key=True)
    agent_class = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    floors_reached = Column(Integer, nullable=False)
    bosses_defeated = Column(Integer, nullable=False)
    win = Column(Boolean, nullable=False)

    # Ensure this matches the relationship
    card_picks = relationship('CardPicked', back_populates='game', cascade="all, delete-orphan")

class CardPicked(Base):
    __tablename__ = 'cards_picked'

    pick_id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.game_id'), nullable=False)
    card_name = Column(String, nullable=False)
    card_id = Column(String, nullable=False)
    other_options = Column(String, nullable=True)
    agent_class = Column(String, nullable=False)

    game = relationship('Game', back_populates='card_picks')

class CardPerformance(Base):
    __tablename__ = 'card_performance'

    card_id = Column(String, primary_key=True)
    card_name = Column(String, nullable=False)
    times_picked = Column(Integer, default=0)
    average_floor_reached = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    games_featured_in = Column(Integer, default=0)

    game = relationship('Game') 
