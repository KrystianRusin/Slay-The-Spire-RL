from keras.models import Model
from keras.layers import Dense, Flatten, Input, Concatenate, LSTM
def build_model(observation_space, nb_actions):
    # Define input layers for each part of the observation space
    player_input = Input(shape=(4,), name='player_input')  # current_hp, max_hp, block, energy
    
    # Define input and processing for cards
    card_input_shape = (10, 9)  # max 10 cards in hand, each card has 9 features
    card_input = Input(shape=card_input_shape, name='card_input')
    card_flatten = Flatten()(card_input)
    
    # Define input and processing for sequences (deck, draw_pile, etc.)
    deck_input_shape = (None, 9)  # variable size deck, each card has 9 features
    deck_input = Input(shape=deck_input_shape, name='deck_input')
    deck_lstm = LSTM(64)(deck_input)  # Process variable size deck with LSTM
    
    # Define input and processing for monsters
    monster_input_shape = (5, 10)  # max 5 monsters, each monster has 10 features
    monster_input = Input(shape=monster_input_shape, name='monster_input')
    monster_flatten = Flatten()(monster_input)
    
    # Define input for screen type
    screen_type_input = Input(shape=(1,), name='screen_type_input')
    
    # Concatenate all processed inputs
    concatenated = Concatenate()([player_input, card_flatten, deck_lstm, monster_flatten, screen_type_input])
    
    # Fully connected layers
    dense1 = Dense(256, activation='relu')(concatenated)
    dense2 = Dense(192, activation='relu')(dense1)
    output = Dense(nb_actions, activation='linear')(dense2)
    
    # Create the model
    model = Model(inputs=[player_input, card_input, deck_input, monster_input, screen_type_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model