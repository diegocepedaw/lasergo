import os
import subprocess
from sgfmill import sgf

def get_response_move(filename):
    ''' shell out  and call the gnugo program and give it an in sgf file and an out sgf file'''
    cmd = r'bin\gnugo-3.8\gnugo.exe --infile '+ filename + r' --quiet --outfile sgf_files\response.sgf'

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    p.kill()


    with open(r"sgf_files\response.sgf", "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    board_size = game.get_size()
    root_node = game.get_root()
    moves = []
    for node in game.get_main_sequence():
        moves.append(node.get_move())
    # return last move which is the response from gnugo
    return(moves[-1:][0])