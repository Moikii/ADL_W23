"""
Functions to decide which player won a current play, and how much points the player gets.
"""

from typing import List, Dict, Tuple

card_values_ascending = ["6", "7", "8", "9", "x", "u", "o", "k", "a"]
card_trump_values_ascending = ["6", "7", "8", "x", "o", "k", "a", "9", "u"]


def get_points(
    current_play: List[str],
    trump: str,
    number_of_played_cards: int,
    number_of_players: int,
) -> int:
    """
    Return points in the current play.
    """
    points = 0
    for card in current_play:
        match card[1]:
            case "x":
                points += 10
            case "u":
                if card[0] == trump:
                    points += 20
                else:
                    points += 2
            case "o":
                points += 3
            case "k":
                points += 4
            case "a":
                points += 11
            case "9":
                if card[0] == trump:
                    points += 14
            case _:
                continue
    if number_of_players > (36 - number_of_played_cards):
        points += 5
    return points


def highest_card_pos(current_play: List[str], trump: str) -> int:
    """
    Return the position of the player with the hightest card in the current play.
    """
    highest_card = current_play[0]
    for card in current_play:
        if card[0] == trump:
            if highest_card[0] == trump:
                if card_trump_values_ascending.index(
                    highest_card[1]
                ) < card_trump_values_ascending.index(card[1]):
                    highest_card = card
            else:
                highest_card = card
        elif card[0] == highest_card[0]:
            if card_values_ascending.index(
                highest_card[1]
            ) < card_values_ascending.index(card[1]):
                highest_card = card
    return current_play.index(highest_card)


def add_points_from_play(
    players_dict: Dict[str, int],
    beginning_player: int,
    current_play: List[str],
    trump: str,
    number_of_played_cards: int,
    number_of_players: int,
) -> Tuple[Dict[str, int], int]:
    """
    Return the dictionary with updated scores and the winner of the current play.
    """
    points = get_points(current_play, trump, number_of_played_cards, number_of_players)
    winner = (
        highest_card_pos(current_play, trump) + beginning_player
    ) % number_of_players
    players_dict[f"Player {winner + 1}"] += points
    return players_dict, winner
