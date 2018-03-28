import os
import subprocess


def make_video(input_files, output_filename):
    FNULL = open(os.devnull, 'w')
    command = ('ffmpeg -y -r 15 -i ' +
               input_files +
               ' -pix_fmt yuv420p "' +
               output_filename +
               '"')

    if subprocess.call(
            command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT) != 0:
        raise SystemError('Could not encode video')
