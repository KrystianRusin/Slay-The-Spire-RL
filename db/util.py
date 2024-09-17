from .session import get_db_session
from .models import Card, GameRecord

def add_card(name):
    with get_db_session() as session:
        new_card = Card(name=name)
        session.add(new_card)
        session.commit()

def add_game_record(character_class, floor_reached, win, card):
    with get_db_session() as session:
        game_record = GameRecord(character_class=character_class, floor_reached=floor_reached, win=win, card=card)
        session.add(game_record)
        session.commit()

def get_favorite_cards():
    with get_db_session() as session:
        cards = session.query(Card).order_by(Card.picked_count.desc()).all()
        return cards
