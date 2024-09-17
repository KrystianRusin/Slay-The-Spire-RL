from sqlalchemy.orm import Session
from db.models import CardPicked, CardPerformance
from db.session import SessionLocal
import json
import math

def track_card_pick(game_state, action, game_id):
    """
    Track the card picked by the agent and store it in the database.
    """
    if action.startswith("CHOOSE"):
        try:
            chosen_index = int(action.split()[1])
            cards = game_state["game_state"]["screen_state"].get("cards", [])

            if 0 <= chosen_index < len(cards):
                chosen_card = cards[chosen_index]

                # Collect other card options except the chosen one
                other_options = [card["name"] for i, card in enumerate(cards) if i != chosen_index]
                other_options_json = json.dumps(other_options)  # Convert to JSON for storage

                # Database session
                db: Session = SessionLocal()

                new_card_pick = CardPicked(
                    game_id=game_id,
                    card_name=chosen_card["name"],
                    card_id=chosen_card["id"],
                    other_options=other_options_json,
                    agent_class=game_state["game_state"].get("class")
                )

                db.add(new_card_pick)
                db.commit()
                db.close()

                print(f"Card '{chosen_card['name']}' picked and added to the database with options {other_options}.")
            else:
                print("Chosen index is out of range.")
        except (IndexError, ValueError):
            print("Error parsing chosen card index from action.")

def track_card_performance(game_state, floor_reached, won):
    """
    Track the performance of cards in the deck at the end of each game.
    :param game_state: The game state containing the deck information.
    :param floor_reached: The floor the agent reached in this game.
    :param won: Boolean indicating whether the game was won or lost.
    """
    deck = game_state.get("deck", [])
    card_counts = {}
    
    # Count occurrences of each card in the deck, excluding "Strike" and "Defend"
    for card in deck:
        if card['name'] in ["Strike", "Defend"]:
            continue
        
        card_id = card['id']
        if card_id not in card_counts:
            card_counts[card_id] = {
                "name": card['name'],
                "count": 1
            }
        else:
            card_counts[card_id]["count"] += 1

    db = SessionLocal()
    try:
        for card_id, card_info in card_counts.items():
            card_name = card_info["name"]
            count = card_info["count"]

            # Retrieve or create the card performance entry
            card_performance = db.query(CardPerformance).filter(CardPerformance.card_id == card_id).first()
            
            if card_performance:
                # Update existing entry
                card_performance.times_picked += count
                card_performance.games_featured_in += 1

                # Update the win rate and average floor reached
                total_games = card_performance.games_featured_in
                card_performance.average_floor_reached = math.floor(
                    ((card_performance.average_floor_reached * (total_games - 1)) + floor_reached) / total_games
                )
                card_performance.win_rate = (((card_performance.win_rate * (total_games - 1)) + (1 if won else 0)) / total_games) * 100
            else:
                # Create a new entry
                card_performance = CardPerformance(
                    card_id=card_id,
                    card_name=card_name,
                    times_picked=count,
                    games_featured_in=1,
                    average_floor_reached=floor_reached,
                    win_rate=1.0 if won else 0.0
                )
                db.add(card_performance)

        db.commit()
    except Exception as e:
        print(f"Error updating card performance: {e}")
        db.rollback()
    finally:
        db.close()

