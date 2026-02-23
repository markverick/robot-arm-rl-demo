import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from .env import Figure8TrackEnv

MODEL_PATH = "ppo_figure8_panda_large_fit_v2.zip"


def draw_ee_trail(viewer, points, radius=0.004):
    if not points:
        return

    max_geoms = len(viewer.user_scn.geoms)
    n = min(len(points), max_geoms)
    start = len(points) - n

    viewer.user_scn.ngeom = 0
    mat = np.eye(3, dtype=np.float64).ravel()

    for i in range(n):
        p = points[start + i]
        alpha = 0.15 + 0.80 * ((i + 1) / n)
        rgba = np.array([0.10, 0.95, 0.35, alpha], dtype=np.float32)
        size = np.array([radius, 0.0, 0.0], dtype=np.float64)

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size,
            pos=p,
            mat=mat,
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


def main():
    # Use the same env params you trained with
    env = Figure8TrackEnv(
        xml_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        ee_body="hand",
        control_hz=40,
        episode_seconds=10.0,
        f_hz=0.75,
        a=0.22,
        b=0.17,
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

    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    ee_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.ee_body)
    if ee_body_id < 0:
        raise ValueError(f"EE body not found: {env.ee_body}")

    pos_errs = []
    ori_errs = []
    rewards = []
    ee_trail = []
    trail_stride = 2
    trail_max_points = 300

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            pos_errs.append(info["pos_err"])
            ori_errs.append(info["ori_err"])
            rewards.append(reward)

            if step % 10 == 0:
                print(
                    f"step={step:4d}  pos_err={info['pos_err']:.4f} m  "
                    f"ori_err={info['ori_err']:.4f} rad  reward={reward:.4f}"
                )

            if step % trail_stride == 0:
                ee_trail.append(env.data.xpos[ee_body_id].copy())
                if len(ee_trail) > trail_max_points:
                    ee_trail = ee_trail[-trail_max_points:]

            with viewer.lock():
                draw_ee_trail(viewer, ee_trail)

            viewer.sync()
            # match sim pace (optional; remove if you want it as fast as possible)
            time.sleep(env.dt)

            step += 1
            if terminated or truncated:
                break

    pos_errs = np.array(pos_errs)
    ori_errs = np.array(ori_errs)
    rewards = np.array(rewards)

    def rms(x): return float(np.sqrt(np.mean(x * x)))

    print("\nEpisode summary:")
    print(f"  steps: {len(pos_errs)}")
    print(f"  pos_err: mean={pos_errs.mean():.4f} m, rms={rms(pos_errs):.4f} m, max={pos_errs.max():.4f} m")
    print(f"  ori_err: mean={ori_errs.mean():.4f} rad, rms={rms(ori_errs):.4f} rad, max={ori_errs.max():.4f} rad")
    print(f"  reward:  mean={rewards.mean():.4f}, sum={rewards.sum():.4f}")

if __name__ == "__main__":
    main()
