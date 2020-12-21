from sgfmill import sgf
game = sgf.Sgf_game(size=19)
for move_info in ...:
    node = game.extend_main_sequence()
    node.set_move(move_info.colour, move_info.move)
    if move_info.comment is not None:
        node.set("C", move_info.comment)
with open("gamestate.sfg", "wb") as f:
    f.write(game.serialise())


def to_SGF(board):
  # Return an SGF representation of the board state
  board_letters = string.ascii_lowercase # 'a' to 'z'
  output = "(;GM[1]FF[4]SZ[" + str(BOARD_SIZE) + "]\n"
  if side_to_move.get() == 1:
    output += "PL[B]\n"
  else:
    output += "PL[W]\n"
  black_moves, white_moves = "", ""
  if BoardStates.BLACK in board:
    black_moves += "AB"
    for i in range(hsize):
      for j in range(vsize):
        if board[i,j] == BoardStates.BLACK:
          black_moves += "[" + board_letters[i] + board_letters[j] + "]"
  if BoardStates.WHITE in board:
    white_moves += "AW"
    for i in range(hsize):
      for j in range(vsize):
        if board[i,j] == BoardStates.WHITE:
          white_moves += "[" + board_letters[i] + board_letters[j] + "]"
  if side_to_move.get() == 1:
    output += black_moves + "\n" + white_moves + "\n" + ")\n"
  else:
    output += white_moves + "\n" + black_moves + "\n" + ")\n"
  # According to the SGF standard, it shouldn't make a difference
  # which order the AB[] and AW[] tags come in,
  # but at the time of writing,
  # Lizzie uses this to deduce which side is to move (ignoring the PL[] tag)!
  return output


def save_SGF():
  global output_file
  if output_file is not None:
    output_file = filedialog.asksaveasfilename(initialfile = output_file)
  else:
    output_file = filedialog.asksaveasfilename()
  sgf = open(output_file, "w")
  sgf.write(to_SGF(full_board))
  sgf.close()
  log("Saved to file " + output_file)