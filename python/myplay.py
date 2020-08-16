#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import re
import logging
import colorsys
import json
import tensorflow as tf
import numpy as np

from board import Board
from model import Model
import common


saved_model_dir = "../saved_modelg170-15"
(model_variables_prefix, model_config_json) = (
    os.path.join(saved_model_dir, "variables", "variables"),
    os.path.join(saved_model_dir, "model.config.json"),
)
name_scope = "swa_model"

# Hardcoded max board size
pos_len = 19

# Model ----------------------------------------------------------------

with open(model_config_json) as f:
    model_config = json.load(f)

with tf.compat.v1.variable_scope(name_scope):
    model = Model(model_config, pos_len, {})


policy0_output = tf.nn.softmax(model.policy_output[:, :, 0])
lead_output = 20.0 * model.miscvalues_output[:, 2]
ownership_output = tf.tanh(model.ownership_output)

class GameState:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = Board(size=board_size)
        self.moves = []
        self.boards = [self.board.copy()]


# Moves ----------------------------------------------------------------


def fetch_output(session, gs, rules, fetches):
    bin_input_data = np.zeros(shape=[1] + model.bin_input_shape, dtype=np.float32)
    global_input_data = np.zeros(shape=[1] + model.global_input_shape, dtype=np.float32)
    pla = gs.board.pla
    opp = Board.get_opp(pla)
    move_idx = len(gs.moves)
    model.fill_row_features(gs.board, pla, opp, gs.boards, gs.moves, move_idx, rules, bin_input_data, global_input_data, idx=0)
    outputs = session.run(
        fetches,
        feed_dict={
            model.bin_inputs: bin_input_data,
            model.global_inputs: global_input_data,
            model.symmetries: [False, False, False],
            model.include_history: [[1.0, 1.0, 1.0, 1.0, 1.0]],
        },
    )
    return [output[0] for output in outputs]


def get_outputs(session, gs, rules):
    fetches = [
        policy0_output,
        lead_output,
        ownership_output,
    ]
    policy0, lead, ownership = fetch_output(session, gs, rules, fetches)
    board = gs.board

    moves_and_probs0 = []
    for i in range(len(policy0)):
        move = model.tensor_pos_to_loc(i, board)
        if i == len(policy0) - 1:
            moves_and_probs0.append((Board.PASS_LOC, policy0[i]))
        elif board.would_be_legal(board.pla, move):
            moves_and_probs0.append((move, policy0[i]))

    ownership_flat = ownership.reshape([model.pos_len * model.pos_len])
    ownership_by_loc = []
    board = gs.board
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            pos = model.loc_to_tensor_pos(loc, board)
            if board.pla == Board.WHITE:
                ownership_by_loc.append((loc, ownership_flat[pos]))
            else:
                ownership_by_loc.append((loc, -ownership_flat[pos]))

    moves_and_probs = sorted(moves_and_probs0, key=lambda moveandprob: moveandprob[1], reverse=True)
    # Generate a random number biased small and then find the appropriate move to make
    # Interpolate from moving uniformly to choosing from the triangular distribution
    alpha = 1
    beta = 1 + math.sqrt(max(0, len(gs.moves) - 20))
    r = np.random.beta(alpha, beta)
    probsum = 0.0
    i = 0
    genmove_result = Board.PASS_LOC
    while i < len(moves_and_probs):
        (move, prob) = moves_and_probs[i]
        probsum += prob
        if i >= len(moves_and_probs) - 1 or probsum > r:
            genmove_result = move
            break
        i += 1

    return {
        "policy0": policy0,
        "moves_and_probs0": moves_and_probs0,
        "lead": lead,
        "ownership_by_loc": ownership_by_loc,
        "genmove_result": genmove_result,
    }


def get_layer_values(session, gs, rules, layer, channel):
    board = gs.board
    [layer] = fetch_output(session, gs, rules=rules, fetches=[layer])
    layer = layer.reshape([model.pos_len * model.pos_len, -1])
    locs_and_values = []
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            pos = model.loc_to_tensor_pos(loc, board)
            locs_and_values.append((loc, layer[pos, channel]))
    return locs_and_values


def get_input_feature(gs, rules, feature_idx):
    board = gs.board
    bin_input_data = np.zeros(shape=[1] + model.bin_input_shape, dtype=np.float32)
    global_input_data = np.zeros(shape=[1] + model.global_input_shape, dtype=np.float32)
    pla = board.pla
    opp = Board.get_opp(pla)
    move_idx = len(gs.moves)
    model.fill_row_features(board, pla, opp, gs.boards, gs.moves, move_idx, rules, bin_input_data, global_input_data, idx=0)

    locs_and_values = []
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            pos = model.loc_to_tensor_pos(loc, board)
            locs_and_values.append((loc, bin_input_data[0, pos, feature_idx]))
    return locs_and_values


def get_pass_alive(board, rules):
    pla = board.pla
    opp = Board.get_opp(pla)
    area = [-1 for i in range(board.arrsize)]
    nonPassAliveStones = False
    safeBigTerritories = True
    unsafeBigTerritories = False
    board.calculateArea(area, nonPassAliveStones, safeBigTerritories, unsafeBigTerritories, rules["multiStoneSuicideLegal"])

    locs_and_values = []
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            locs_and_values.append((loc, area[loc]))
    return locs_and_values


def print_scorebelief(gs, outputs):
    board = gs.board
    scorebelief = outputs["scorebelief"]
    scoremean = outputs["scoremean"]
    scorestdev = outputs["scorestdev"]
    sbscale = outputs["sbscale"]

    scorebelief = list(scorebelief)
    if board.pla != Board.WHITE:
        scorebelief.reverse()
        scoremean = -scoremean

    scoredistrmid = pos_len * pos_len + Model.EXTRA_SCORE_DISTR_RADIUS
    ret = ""
    ret += "TEXT "
    ret += "SBScale: " + str(sbscale) + "\n"
    ret += "ScoreBelief: \n"
    for i in range(17, -1, -1):
        ret += "TEXT "
        ret += "%+6.1f" % (-(i * 20 + 0.5))
        for j in range(20):
            idx = scoredistrmid - (i * 20 + j) - 1
            ret += " %4.0f" % (scorebelief[idx] * 10000)
        ret += "\n"
    for i in range(18):
        ret += "TEXT "
        ret += "%+6.1f" % ((i * 20 + 0.5))
        for j in range(20):
            idx = scoredistrmid + (i * 20 + j)
            ret += " %4.0f" % (scorebelief[idx] * 10000)
        ret += "\n"

    beliefscore = 0
    beliefscoresq = 0
    beliefwin = 0
    belieftotal = 0
    for idx in range(scoredistrmid * 2):
        score = idx - scoredistrmid + 0.5
        if score > 0:
            beliefwin += scorebelief[idx]
        else:
            beliefwin -= scorebelief[idx]
        belieftotal += scorebelief[idx]
        beliefscore += score * scorebelief[idx]
        beliefscoresq += score * score * scorebelief[idx]

    beliefscoremean = beliefscore / belieftotal
    beliefscoremeansq = beliefscoresq / belieftotal
    beliefscorevar = max(0, beliefscoremeansq - beliefscoremean * beliefscoremean)
    beliefscorestdev = math.sqrt(beliefscorevar)

    ret += "TEXT BeliefWin: %.2fc\n" % (100 * beliefwin / belieftotal)
    ret += "TEXT BeliefScoreMean: %.1f\n" % (beliefscoremean)
    ret += "TEXT BeliefScoreStdev: %.1f\n" % (beliefscorestdev)
    ret += "TEXT ScoreMean: %.1f\n" % (scoremean)
    ret += "TEXT ScoreStdev: %.1f\n" % (scorestdev)
    return ret


# Basic parsing --------------------------------------------------------
colstr = "ABCDEFGHJKLMNOPQRST"


def parse_coord(s, board):
    if s == "pass":
        return Board.PASS_LOC
    return board.loc(colstr.index(s[0].upper()), board.size - int(s[1:]))


def str_coord(loc, board):
    if loc == Board.PASS_LOC:
        return "pass"
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    return "%c%d" % (colstr[x], board.size - y)


# GTP Implementation -----------------------------------------------------

def run_gtp(session):
    board_size = 19
    gs = GameState(board_size)

    rules = {
        "koRule": "KO_POSITIONAL",
        "scoringRule": "SCORING_AREA",
        "taxRule": "TAX_NONE",
        "multiStoneSuicideLegal": True,
        "hasButton": False,
        "encorePhase": 0,
        "passWouldEndPhase": False,
        "whiteKomi": 7.5,
    }

    input_feature_command_lookup = dict()

    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if line == "":
            continue
        command = [s.lower() for s in line.split()]
        if re.match(r"\d+", command[0]):
            cmdid = command[0]
            command = command[1:]
        else:
            cmdid = ""

        ret = ""
        if command[0] == "boardsize":
            if int(command[1]) > model.pos_len:
                print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
                ret = None
            board_size = int(command[1])
            gs = GameState(board_size)
        elif command[0] == "clear_board":
            gs = GameState(board_size)
        elif command[0] == "showboard":
            ret = "\n" + gs.board.to_string().strip()
        elif command[0] == "komi":
            rules["whiteKomi"] = float(command[1])
        elif command[0] == "play":
            pla = Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE
            loc = parse_coord(command[2], gs.board)
            gs.board.play(pla, loc)
            gs.moves.append((pla, loc))
            gs.boards.append(gs.board.copy())
        elif command[0] == "genmove":
            outputs = get_outputs(session, gs, rules)
            loc = outputs["genmove_result"]
            pla = gs.board.pla

            if len(command) > 1:
                pla = Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE
            gs.board.play(pla, loc)
            gs.moves.append((pla, loc))
            gs.boards.append(gs.board.copy())
            ret = str_coord(loc, gs.board)
        else:
            print("Warning: Ignoring unknown command - %s" % (line,), file=sys.stderr)
            ret = None

        if ret is not None:
            print("=%s %s\n\n" % (cmdid, ret,), end="")
        else:
            print("?%s ???\n\n" % (cmdid,), end="")
        sys.stdout.flush()


saver = tf.train.Saver(max_to_keep=10000, save_relative_paths=True,)

with tf.Session() as session:
    saver.restore(session, model_variables_prefix)
    run_gtp(session)
