"""Lanza corridas separadas y reanudables de PPO y recompensa."""

import argparse

from comun import OUT, preparar_notebook


def main():
    preparar_notebook()
    import torch

    from rlhf import (
        PPO_EXPERIMENTS, REWARD_6E_DIR, clear_gpu, confirm_reward, read_partitions, train_ppo, train_reward,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrida", choices=[*PPO_EXPERIMENTS, "recompensa"])
    parser.add_argument("--piloto", action="store_true")
    parser.add_argument("--pasadas", type=int, choices=[1, 2], default=1)
    parser.add_argument("--microlote", type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument("--lote-generacion", type=int, choices=[2, 4, 8], default=8)
    parser.add_argument("--beta-kl", type=float, choices=[0.02, 0.05, 0.1], default=0.05)
    arguments = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    parts, hashes = read_partitions(arguments.piloto, expanded=True)
    if arguments.corrida == "recompensa":
        path, report = train_reward(parts, hashes, arguments.piloto, expanded=True)
        clear_gpu()
        if not arguments.piloto:
            confirmation = confirm_reward(parts, hashes, path, report)
            print("Confirmación:", confirmation["criterios"], flush=True)
    else:
        reference, reference_hashes = read_partitions(False)
        reward_path, reward = train_reward(reference, reference_hashes, False, six_epochs=True)
        if not (OUT / REWARD_6E_DIR / "confirmacion.json").is_file():
            confirm_reward(reference, reference_hashes, reward_path, reward)
        clear_gpu()
        path, report = train_ppo(
            parts, hashes, reward_path, reward, arguments.piloto,
            experiment=arguments.corrida, passes=arguments.pasadas, microbatch=arguments.microlote,
            generation_batch=arguments.lote_generacion, beta=arguments.beta_kl,
        )
        print(f"Lotes PPO: {len(report['historial'])}", flush=True)
    print(f"Salida: {path.name}", flush=True)


if __name__ == "__main__":
    main()
