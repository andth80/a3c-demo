import threading
from time import sleep

import yappi

import main


def profile(rom_name, checkpoint):
    thread = threading.Thread(target=main.main, args=[rom_name, checkpoint])
    thread.start()

    yappi.start()

    sleep(300)

    yappi.get_func_stats().print_all()


if __name__ == '__main__':
    rom_name, checkpoint = main.get_commandline_args()
    profile(rom_name, checkpoint)
