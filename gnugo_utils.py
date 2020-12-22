# from subprocess import Popen, PIPE
# output = Popen([r'C:\Users\diego\Documents\gnugo-3.8\gnugo.exe', '--infile', r'C:\Users\diego\Documents\lasergo\sgf_out.sgf'], stdout=PIPE)
# a = pipe.readline()
# print(a)
# print(output.stdout.read())
# output = output.communicate()[0]
# print(output)
# cmdCommand = r"C:\Users\diego\Documents\gnugo-3.8\gnugo.exe --infile C:\Users\diego\Documents\lasergo\sgf_out.sgf"   #specify your cmd command
# process = subprocess.Popen(cmdCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
# print(output)


# from subprocess import check_output
# result = check_output(r"C:\Users\diego\Documents\gnugo-3.8\gnugo.exe --infile C:\Users\diego\Documents\lasergo\sgf_out.sgf", shell=True)


import os
import subprocess
from sgfmill import sgf

def get_response_move(filename):
    cmd = r'C:\Users\diego\Documents\gnugo-3.8\gnugo.exe --infile '+ filename + r' --quiet --outfile C:\Users\diego\Documents\lasergo\sgf_files\response.sgf'

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    #print(out)
    p.kill()


    with open(r"C:\Users\diego\Documents\lasergo\sgf_files\response.sgf", "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    #winner = game.get_winner()
    board_size = game.get_size()
    root_node = game.get_root()
    moves = []
    for node in game.get_main_sequence():
        moves.append(node.get_move())
    # return last move which is the response from gnugo
    return(moves[-1:][0])