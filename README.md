# Install

## Ubuntu 22.04/24.04

Install required system packages and rendering libraries:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
sudo apt-get install -y libglfw3 libglew-dev libgl1-mesa-glx libosmesa6
```

Create and activate a clean Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install MuJoCo + Gymnasium + Stable-Baselines3:

```bash
pip install mujoco gymnasium "stable-baselines3[extra]"
```

Clone MuJoCo Menagerie models into this repo:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Run everything automatically:

```bash
bash setup.sh
```
