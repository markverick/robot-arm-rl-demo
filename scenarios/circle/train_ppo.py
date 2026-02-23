from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
from datetime import datetime

from .env import CircleTrackEnv


def make_env():
    env = CircleTrackEnv(
        xml_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        ee_body="hand",
        control_hz=40,
        episode_seconds=10.0,
        radius=0.20,
        f_hz=0.7,
        alpha=0.08,
        theta_max_deg=6.0,
        action_scale=0.03,
        terminate_pos_err_m=1.20,
        termination_grace_steps=25,
        reward_pos_w=25.0,
        reward_ori_w=1.0,
        reward_act_w=0.02,
        seed=0,
    )
    return Monitor(env)


def main():
    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        verbose=1,
        n_steps=4096,
        batch_size=512,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=2e-4,
        clip_range=0.15,
        ent_coef=0.0,
    )

    model.learn(total_timesteps=800_000)
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_stem = model_dir / f"ppo_circle_panda_{timestamp}"
    model.save(str(model_path_stem))
    print(f"Saved to {model_path_stem}.zip")


if __name__ == "__main__":
    main()
