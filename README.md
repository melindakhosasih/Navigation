# Navigation

## Installation
Tested in Python 3.11.4

- conda

        conda env create -f nav.yaml
        conda activate nav

- pip

        pip install -r requirements.txt


## Wrapper Test

- Random Action

        python wrapper.py --mode 0

- Control Manually

        python wrapper.py --mode 1

    - <kbd>W</kbd> : Move Forward
    - <kbd>A</kbd> : Turn Left
    - <kbd>D</kbd> : Turn Right
    - <kbd>R</kbd> : Reset
    - <kbd>Esc</kbd> : Exit

## Train
- Train ddpg

        python train.py --title test --algo ddpg

- Train sac

        python train.py --title test --algo sac

## Evaluation

- Evaluate ddpg

        python eval.py --result_name demo_ddpg --model_dir out/ddpg/test/ --algo ddpg

- Evaluate sac

        python eval.py --result_name demo_sac --model_dir out/sac/test/ --algo sac

## Demo

- DDPG

    <img src="demo/demo_ddpg.gif">

- SAC

    <img src="demo/demo_sac.gif">