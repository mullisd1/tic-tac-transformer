import os

import torch
import numpy as np

from tictactoe.board import Board
from tictactransformer.utils.gather_data import boards_to_format
from tictactransformer.models.transformer import TicTacToeModel


def clear():
    """
    :info: clears the output
    """
    os.system("clear")


def clear_lines(lines: int):
    """
    :info: clears last (n) lines of output
    :param lines: number of lines to clear
    """
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for _ in range(lines):
        print(LINE_UP, end=LINE_CLEAR)


def get_size() -> int:
    ### Get Board Size
    size = 0
    while size == 0:
        size = int(input(f"Enter Board Size: "))
        if size < 3 or size > 10:
            print(f"{size} is not a valid size")
            size = 0
    clear()
    return size


def game(size: int) -> str:
    """
    :info: tic tac toe game for a (n) sized board
    :param size: size of the board
    :return: winner
    """
    ### Init Board
    board = Board(size)
    board.print_board()

    model = load_model()

    ### Get Next Game
    model_turn = None
    while model_turn is None:
        an = input(f"Model starts first?: Y/N")
        an = an.capitalize()
        if an != "Y" and an != "N":
            print(f"{an} is not a valid input. Try again")
        else:
            model_turn = (an == "Y")


    ### Play Game
    while board.check_board() == "-":

        if model_turn:
            board_state = boards_to_format(board.to_str())
            
            # print(f"Board State: {board_state}")
            next_board_state = model.forward(torch.tensor([board_state]))
            output = torch.argmax(next_board_state, dim=2)
            output = output.numpy() - 1
            output = output.reshape(3, 3)

            num_diff = np.sum(board.board != output)
            if num_diff == 1:
                board.set_board(output)
            else:
                print(f"Current state: {board.board}")
                print(f"Next state: {output}")
                raise ValueError(f"Model returned invalid board state")
            model_turn = not model_turn
            clear()
            board.print_board()

        else:
            print(f"{board.get_turn()}'s turn")
            row = int(input(f"Input Row: "))
            col = int(input(f"Input Col: "))

            try:
                board.place(row, col)
            except ValueError:
                num_inputs = 3
                clear_lines(num_inputs)
                print(f"Row: {row}, Col: {col} is not a valid position")
            else:
                clear()
                board.print_board()
            model_turn = not model_turn

    ### End Game
    clear()
    board.print_board()

    winner = board.check_board()
    if winner == "X" or winner == "O":
        print(f"Winner {winner}")
    else:
        print(winner)
    return winner

def load_model():
    model = TicTacToeModel(vocab_size = 3,
                           num_embedding = 32,
                           num_heads = 8,
                           num_blocks = 4,
                           block_size = 9,
                           dropout = 0.0
                           )
    # model_path = "./tictactransformer/models/weights/best_model.pth"
    # model_path = "./models/exp1/best_model.pth"
    # model_path = "./models/exp2/best_model.pth"
    model_path = "./models/exp3/best_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def tictactoe():
    """
    :info: game size/repeat/output flow
    """
    clear()
    print(f"Welcome to Tic-Tac-Toe")

    size = 3
    game_num = 1
    another = "Y"
    while another == "Y":
        winner = game(size)

        ### Get Next Game
        done = False
        while not done:
            an = input(f"Play Again? (Y/N): ")
            an = an.capitalize()
            if an != "Y" and an != "N":
                print(f"{an} is not a valid input. Try again")
            else:
                clear()
                another = an
                game_num += 1
                done = True    
    
    


if __name__ == '__main__':
    tictactoe()