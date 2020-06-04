ML-demos
===

### GPU acceleration

```
$ dpkg -l | grep cuda
$ dpkg -l | grep TensorRT
$ nvidia-smi  # Show current GPU processes using VRAM
```

#### CUDA libs:
```
CUDA 10.2 (libcudart 10.2)    [CUDA Runtime]  (/usr/local/cuda-10.2/targets/x86_64-linux/lib)
CUPTI 10.2 (libcupti 10.2)    [CUDA profiling tools]  (/usr/local/cuda-10.2/extras/CUPTI/lib64)
cuDNN 7.6.5 (libcudnn 7.6.5)  [CUDA Deep Neural Network]  (/usr/lib/x86_64-linux-gnu)
TensorRT 7.0.0 (libnvinfer, graphsurgeon-tf, python3-libnvinfer)  (/usr/lib/x86_64-linux-gnu)
```

### TensorBoard Metrics
```
$ tensorboard --logdir=logs/
TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

### CARLA
Run CARLA server
```
./CarlaUE4.sh -quality-level=Low
```

#### Dynamic Weather
```
python PythonAPI/examples/dynamic_weather.py
```

#### Spawn NPCs
```
python PythonAPI/examples/spawn_npc.py -n 200
```

#### Manual Control

```
$ python PythonAPI/examples/manual_control.py 

Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit

```