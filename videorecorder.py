import os
import shutil
import PIL

import ffmpeg


class VideoRecorder():

    def __init__(self, recordings_directory):
        self.current_recording_name = None
        self._init_recordings_dir(recordings_directory)
        self._init_recording_tmp_dir()

    def _init_recordings_dir(self, recordings_directory):
        self.recordings_dir = recordings_directory
        if not os.path.exists(recordings_directory):
            os.makedirs(recordings_directory)

    def _init_recording_tmp_dir(self):
        self.recordings_tmp_dir = os.path.join(self.recordings_dir, 'tmp')
        if os.path.exists(self.recordings_tmp_dir):
            shutil.rmtree(self.recordings_tmp_dir)

        os.makedirs(self.recordings_tmp_dir)

    def start_recording(self, recording_name):
        if self.current_recording_name is not None:
            raise RuntimeError('Already recording')

        self.current_recording_name = recording_name
        self.frame_count = 0
        self._init_recording_tmp_dir()

    def stop_recording(self):
        if self.current_recording_name is None:
            raise RuntimeError('Not recording')

        input_files = os.path.join(self.recordings_tmp_dir, '%06d.png')
        output_filename = os.path.join(
            self.recordings_dir, self.current_recording_name + '.mp4')
        ffmpeg.make_video(input_files, output_filename)

        self.current_recording_name = None

    def add_frame(self, frame):
        if self.current_recording_name is None:
            raise RuntimeError('Not recording')

        filename = '{0:06d}.png'.format(self.frame_count)
        file_path = os.path.join(self.recordings_tmp_dir, filename)

        img = PIL.Image.fromarray(frame)
        img.save(file_path)

        self.frame_count += 1

    def reset(self):
        self._init_recording_tmp_dir()
