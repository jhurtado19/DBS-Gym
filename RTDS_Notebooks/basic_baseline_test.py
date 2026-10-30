"""Small smoke test: MLP-PPO vs no DBS vs constant high-frequency DBS.

This is intentionally a fast sanity check, not a scientific benchmark. It uses
env0, a shortened episode, one training seed, and a small PPO budget.
Run it from the repository root after installing requirements_pip.txt.
"""

from copy import deepcopy

import numpy as np
from stable_baselines3 import PPO

from environment.env import SpatialKuramoto
from environment.env_configs.env0 import (
    coord_modif,
    grid_size,
    locus_center,
    locus_size,
    n_neurons,
    params_dict_train,
)
from environment.utils import calc_beta_band_power, generate_w0_with_locus, units2sec


SEED = 228
PPO_STEPS = 256
EPISODES = 2


def make_params():
    params = deepcopy(params_dict_train)
    params.update(
        reward_func="bbpow_action",
        # Short settings for a quick smoke test.
        transient_state_len=20.0,
        observe_wind_counts=20,
        total_episode_len=90.0,
        verbose=0,
        rand_seed=SEED,
    )

    w0, neur_coords, neur_grid, w0_without_locus, w_locus, locus_mask = (
        generate_w0_with_locus(
            n_neurons,
            grid_size,
            coord_modif,
            locus_center=locus_center,
            locus_size=locus_size,
            wmuL=params["wmuL"],
            wsdL=params["wsdL"],
            show=False,
        )
    )
    params.update(
        w0=w0,
        w0_without_locus=w0_without_locus,
        locus_without_w0=w_locus,
        locus_mask=locus_mask,
        neur_coords=neur_coords,
        neur_grid=neur_grid,
    )
    return params


def make_env(params):
    return SpatialKuramoto(params_dict=deepcopy(params))


def evaluate_constant(env, normalized_action, episodes):
    rewards = []
    energies = []
    beta_powers = []
    dt_seconds = units2sec(env.params_dict["verbose_dt"])

    for episode in range(episodes):
        # reset() uses NumPy's global RNG, so reseed to pair episodes across
        # controllers as closely as possible.
        np.random.seed(SEED + episode)
        env.reset()
        episode_reward = 0.0
        episode_energy = 0.0
        lfp = []
        terminated = False

        while not terminated:
            _, reward, terminated, truncated, _ = env.step(
                np.asarray([normalized_action], dtype=np.float32)
            )
            episode_reward += float(reward)
            episode_energy += float(np.abs(env.u).sum())
            lfp.append(env.theta_mean)
            if truncated:
                break

        signal = np.concatenate(lfp)
        beta_power = calc_beta_band_power(signal, dt_seconds, 12.5, 21.0)
        rewards.append(episode_reward)
        energies.append(episode_energy)
        beta_powers.append(beta_power)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_sd": float(np.std(rewards, ddof=1)) if episodes > 1 else 0.0,
        "energy_mean": float(np.mean(energies)),
        "beta_power_mean": float(np.mean(beta_powers)),
    }


def evaluate_policy(model, env, episodes):
    rewards = []
    energies = []
    beta_powers = []
    dt_seconds = units2sec(env.params_dict["verbose_dt"])

    for episode in range(episodes):
        np.random.seed(SEED + episode)
        observation, _ = env.reset()
        episode_reward = 0.0
        episode_energy = 0.0
        lfp = []
        terminated = False

        while not terminated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            episode_energy += float(np.abs(env.u).sum())
            lfp.append(env.theta_mean)
            if truncated:
                break

        signal = np.concatenate(lfp)
        beta_power = calc_beta_band_power(signal, dt_seconds, 12.5, 21.0)
        rewards.append(episode_reward)
        energies.append(episode_energy)
        beta_powers.append(beta_power)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_sd": float(np.std(rewards, ddof=1)) if episodes > 1 else 0.0,
        "energy_mean": float(np.mean(energies)),
        "beta_power_mean": float(np.mean(beta_powers)),
    }


def main():
    np.random.seed(SEED)
    params = make_params()

    train_env = make_env(params)
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=64,
        batch_size=64,
        seed=SEED,
        verbose=1,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=PPO_STEPS, progress_bar=True)

    results = {
        "MLP-PPO": evaluate_policy(model, make_env(params), EPISODES),
        "No DBS": evaluate_constant(make_env(params), 0.0, EPISODES),
        "HF-DBS/cDBS": evaluate_constant(make_env(params), 1.0, EPISODES),
    }

    print("\nController comparison")
    print("controller,reward_mean,reward_sd,energy_mean,beta_power_mean")
    for name, metrics in results.items():
        print(
            f"{name},{metrics['reward_mean']:.6g},{metrics['reward_sd']:.6g},"
            f"{metrics['energy_mean']:.6g},{metrics['beta_power_mean']:.6g}"
        )


if __name__ == "__main__":
    main()
