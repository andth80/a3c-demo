Implementation of the A3C algorithm described in this paper: https://arxiv.org/pdf/1602.01783.pdf. I've tried to follow the naming conventions used in the paper to make it clearer to follow (e.g. used "theta" as the name for the combined set of variables representing the policy and value functions).

## Install and run

### Install dependencies

install ffmpeg:

```
apt-get install ffmpeg
```

install tensorflow:

```
pip install tensorflow
```

install pillow:

```
pip install pillow-simd
```

install ALE:

https://github.com/mgbellemare/Arcade-Learning-Environment

Download some ROMs for the games.

### Running

run from the command line as follows:

```
python main.py <rom name> [checkpoint]
```

If a checkpoint is specified it will load the initial graph values from the checkpoint file.

Upon running it will start training on the game in the specified ROM file and will continue until terminated. Every 10 minutes or so it will evaluate its progress on a series of games, record the average score achieved and also produce a video of it playing through a single episode.