from PIL import Image
import numpy as np
from random import randrange

from ale_python_interface import ALEInterface

import config


def initialise_s():
    return np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT, config.M])


def _clip(hi, lo, value):
    return lo if value < lo else hi if value > hi else value


def get_phi(s):
    ret = np.zeros(
        (config.INPUT_WIDTH, config.INPUT_HEIGHT, config.M), dtype=np.uint8)
    n = max(len(s) - config.M, 0)
    for x in range(n, len(s)):
        ret[:, :, x - n] = s[x]
    return ret


def preprocess_screen(screen):
    ret = Image.fromarray(screen)
    ret = ret.resize((config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR)
    ret = ret.convert('L')
    return np.asarray(ret, dtype=np.uint8)


class Env():

    def __init__(self, rom_name):
        self.__initALE()
        self.__loadROM(rom_name)
        self.screen_history = []
        self.screens = []

    def __initALE(self):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', randrange(1000))
        self.ale.setInt(b'fragsize', 64)
        self.ale.setInt(b'frame_skip', 1)

        # qq set this back to 0.25?
        self.ale.setFloat(b'repeat_action_probability', 0)
        self.ale.setLoggerMode('error')

    def __loadROM(self, rom_name):
        self.ale.loadROM(rom_name.encode('utf-8'))
        self.actions = self.ale.getMinimalActionSet()

        (width, height) = self.ale.getScreenDims()
        self.screen_data1 = np.empty((height, width, 3), dtype=np.uint8)
        self.screen_data2 = np.empty((height, width, 3), dtype=np.uint8)

    def get_legal_action_count(self):
        return len(self.actions)

    def act(self, action_index):
        action = self.actions[action_index]
        reward = 0

        # perform the action 4 times
        reward += _clip(self.ale.act(action), -1, 1)
        reward += _clip(self.ale.act(action), -1, 1)
        reward += _clip(self.ale.act(action), -1, 1)
        self.ale.getScreenRGB(self.screen_data1)
        reward += _clip(self.ale.act(action), -1, 1)
        self.ale.getScreenRGB(self.screen_data2)

        # return the pixel-wise max of the last two frames (some games only
        # render every other frame)
        screen_data_combined = np.maximum(self.screen_data1, self.screen_data2)
        terminal = self.ale.game_over()

        self.screens.append(preprocess_screen(screen_data_combined))
        phi = get_phi(self.screens)

        return (terminal, reward, phi, self.screen_data2)

    def get_s(self):
        return get_phi(self.screens)

    def reset(self):
        self.ale.reset_game()
        self.screens = []
