from sgfmill import sgf
import pickle
import string

BOARD_SIZE = 19
output_file = 'sgf_out.sgf'

def to_SGF(board):
  # Return an SGF representation of the board state
  side_to_move = "black"
  board_letters = string.ascii_lowercase # 'a' to 'z'
  output = "(;GM[1]FF[4]SZ[" + str(BOARD_SIZE) + "]\n"
  if side_to_move == "black":
    output += "PL[B]\n"
  else:
    output += "PL[W]\n"
  black_moves, white_moves = "", ""
  for i, row in enumerate(board):
    for j, color in enumerate(row):
      if color == "black":
        black_moves += "AB"
        black_moves += "[" + board_letters[i] + board_letters[j] + "]"
      elif color == "white":
        white_moves += "AW"
        white_moves += "[" + board_letters[i] + board_letters[j] + "]"

  if side_to_move == "black":
    output += black_moves + "\n" + white_moves + "\n" + ")\n"
  else:
    output += white_moves + "\n" + black_moves + "\n" + ")\n"
  return output


def save_SGF(board):
  global output_file
  sgf = open(output_file, "w")
  sgf.write(to_SGF(board))
  sgf.close()
  print("Saved to file " + output_file)

if __name__ == "__main__":
  with open('board_state.data', 'rb') as filehandle:
        # read the data as binary data stream
        board = pickle.load(filehandle)
  save_SGF(board)
