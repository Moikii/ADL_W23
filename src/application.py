import requests
import cv2 as cv
import numpy as np
import imutils
import gdown
from ultralytics import YOLO
import streamlit as st
from ultralytics.utils.plotting import Annotator
import jass_rules as jass


model = YOLO('/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/models/large_dataset_best_1.pt')
before_detected_cards = list()
currently_detected_cards = set()
already_played_cards = set()
current_play = list()
game_running = False
beginning_player = 0


if 'disabled' not in st.session_state:
    st.session_state.disabled = False

if 'turned_on' not in st.session_state:
    st.session_state.turned_on = False



st.title('SHEL Demo: Playing card recognition using YOLO')


trump_input = st.selectbox('Trump in this round:',('Schelle', 'Herz', 'Eichel', 'Laub'), disabled = st.session_state.disabled)
trump = trump_input.lower()[0]
number_of_players = st.number_input('Number of players:', 2, 6, step = 1, disabled = st.session_state.disabled)
camera_url_input = st.text_input('Address displayed in IP Webcam:', 'http://192.168.178.39:8080', disabled = st.session_state.disabled)
camera_url = camera_url_input  + '/shot.jpg'
show_camera = st.checkbox('Show camera:', value = st.session_state.turned_on, disabled = st.session_state.disabled)

placeholder = st.empty()
players_dict = dict([(f'Player {i+1}', 0) for i in range(number_of_players)])
placeholder.dataframe(players_dict)

def start_game():
    st.session_state.turned_on = True
    st.session_state.disabled = True

def reset_game():
    st.session_state.disabled = False
    st.session_state.turned_on = False


if st.button('Start Game!', disabled = st.session_state.disabled, on_click = start_game):
    print('Game started!')
    game_running = True


FRAME_WINDOW = st.image([])

if st.button('Reset Game', disabled = not st.session_state.disabled, on_click = reset_game):
    print('Game resetted!')
    before_detected_cards = list()
    currently_detected_cards = set()
    already_played_cards = set()
    current_play = list()
    game_running = False
    beginning_player = 0
    players_dict = dict([(f'Player {i+1}', 0) for i in range(number_of_players)])
    placeholder.dataframe(players_dict)



while show_camera:
    img_resp = requests.get(camera_url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1500, height=2400)
    result = model(source = img, conf = 0.7, verbose = False)[0]
    color_list = [(212,255,127), (87,139,46), (128,0,0), (226,43,138), (143,143,188), (30,105,210), (222,196,176)]
    annotator = Annotator(img)

    for c in result.boxes.cls:
        card = model.names[int(c)]
        currently_detected_cards.add(card)

        if (all((card in frame) for frame in before_detected_cards)) and (card not in already_played_cards) and game_running:
            already_played_cards.add(card)
            current_play.append(card)
            if len(current_play) == number_of_players:
                print(f'Cards from current play (starting with player {beginning_player + 1}): {current_play}')
                players_dict, beginning_player = jass.add_points_from_play(players_dict, beginning_player, current_play,
                                                                            trump, len(already_played_cards), number_of_players)
                placeholder.dataframe(players_dict)
                current_play = list()
            
    if (36 -len(already_played_cards)) < number_of_players:
        st.warning(f'{max(players_dict, key = players_dict.get)} wins!', icon = '🔥')
        print('Game resetted!')
        before_detected_cards = list()
        currently_detected_cards = set()
        already_played_cards = set()
        current_play = list()
        game_running = False
        beginning_player = 0
        players_dict = dict([(f'Player {i+1}', 0) for i in range(number_of_players)])
        placeholder.dataframe(players_dict)
        st.button('Ok', on_click = reset_game())


    if len(before_detected_cards) > 40:
        del before_detected_cards[0]
    before_detected_cards.append(currently_detected_cards)

    for i, label in enumerate(result.boxes.data):
        annotator.box_label(
            label[0:4],
            f"{result.names[label[-1].item()]} {round(label[-2].item(), 2)}", color_list[i%len(color_list)])
    img = cv.cvtColor(annotator.im, cv.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img)