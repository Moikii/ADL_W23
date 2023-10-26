'''
This file contains the code to run a small demo-application using Streamlit. How to launch the
application is described in the README.md.
'''
import os
import requests
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import streamlit as st
import jass_rules as jass


# initialize global variables
game_running = False
before_detected_cards = list()
currently_detected_cards = set()
already_played_cards = set()
current_play = list()
beginning_player = 0



MODEL_PATH = str(os.getcwd()) + '/models/best.pt'
model = YOLO(MODEL_PATH)
#model = YOLO('/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/models/best.pt')


def start_game():
    st.session_state.input_disabled = True
    st.session_state.camera_turned_on = True

def reset_game():
    st.session_state.input_disabled = False
    st.session_state.camera_turned_on = False


# set session_states to later disable inputs/camera on certain events
if 'input_disabled' not in st.session_state:
    st.session_state.input_disabled = False
    st.session_state.camera_turned_on = False
    

# create user interface
st.title('SHEL Demo: Playing card recognition using YOLO')
trump_input = st.selectbox('Trump in this round:',('Schelle', 'Herz', 'Eichel', 'Laub'), disabled = st.session_state.input_disabled)
trump = trump_input.lower()[0]
number_of_players = st.number_input('Number of players:', 2, 6, step = 1, disabled = st.session_state.input_disabled)
camera_url_input = st.text_input('Address displayed in IP Webcam:', 'http://192.168.178.39:8080', disabled = st.session_state.input_disabled)
camera_url = camera_url_input  + '/shot.jpg'
show_camera = st.checkbox('Show camera:', value = st.session_state.camera_turned_on, disabled = st.session_state.input_disabled)
placeholder = st.empty()
players_dict = dict([(f'Player {i+1}', 0) for i in range(number_of_players)])
placeholder.dataframe(players_dict)

if st.button('Start Game!', disabled = st.session_state.input_disabled, on_click = start_game):
    print('Game started!')
    print(f'Trump: {trump_input}')
    game_running = True

live_video = st.image([])

if st.button('Reset Game', disabled = not st.session_state.input_disabled, on_click = reset_game):
    print('Game reset!')
    game_running = False
    before_detected_cards = list()
    currently_detected_cards = set()
    already_played_cards = set()
    current_play = list()
    beginning_player = 0


# loop showing detected playing cards
while show_camera:
    img_from_server = requests.get(camera_url) 
    img_array = np.array(bytearray(img_from_server.content), dtype=np.uint8)
    img = cv.imdecode(img_array, -1)
    # detect cards using YOLO
    result = model(source = img, conf = 0.7, verbose = False)[0]
    # do score calculations when game is going on
    if game_running:
        for card_class in result.boxes.cls:
            card = model.names[int(card_class)]
            currently_detected_cards.add(card)
            # if card was detected consecutively for n frames and was not registered before, count as played card
            if (all((card in frame) for frame in before_detected_cards)) and (card not in already_played_cards):
                already_played_cards.add(card)
                current_play.append(card)
                if len(current_play) == number_of_players:
                    print(f'Cards from current play (starting with player {beginning_player + 1}): {current_play}')
                    players_dict, beginning_player = jass.add_points_from_play(players_dict, beginning_player, current_play, trump,
                                                                               len(already_played_cards), number_of_players)
                    placeholder.dataframe(players_dict)
                    current_play = list()

                    # when last round was played, display winner and end game       
                    if (36 -len(already_played_cards)) < number_of_players:
                        st.warning(f'{max(players_dict, key = players_dict.get)} wins!', icon = '🔥')
                        game_running = False
        
        # keep track of the detected cards in the last 40 frames
        if len(before_detected_cards) > 40:
            del before_detected_cards[0]
        before_detected_cards.append(currently_detected_cards)

    # due to visualizing with Streamlit, output image of model cannot be used and bounding boxes need to be added manually
    color_list = [(212,255,127), (87,139,46), (128,0,0), (226,43,138), (143,143,188), (30,105,210), (222,196,176)]
    annotator = Annotator(img)
    for i, label in enumerate(result.boxes.data):
        annotator.box_label(
            label[0:4],
            f"{result.names[label[-1].item()]} {round(label[-2].item(), 2)}", color_list[i%len(color_list)])
    img = cv.cvtColor(annotator.im, cv.COLOR_BGR2RGB)
    live_video.image(img)