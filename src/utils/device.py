"""
Détection automatique du meilleur device disponible.

Le but : écrire le code UNE fois et l'exécuter partout sans le modifier.
  - en local sur ton Mac      -> MPS (le GPU Apple Silicon)
  - sur vast.ai               -> CUDA (le GPU NVIDIA loué)
  - en dernier recours        -> CPU

Usage dans n'importe quel script :
    from src.utils.device import get_device
    device = get_device()
    model = model.to(device)
    batch = batch.to(device)
"""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """Renvoie le meilleur device disponible, dans l'ordre CUDA > MPS > CPU.

    CUDA passe avant MPS volontairement : quand tu es sur vast.ai, tu veux
    le GPU NVIDIA. MPS n'est choisi que si CUDA est absent (donc en local).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[device] CUDA actif -> {name} ({mem:.0f} Go)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[device] MPS actif -> GPU Apple Silicon (local)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[device] Aucun GPU -> CPU (lent, OK pour debug seulement)")
    return device


def assert_gpu_or_warn() -> None:
    """Garde-fou anti-gaspillage à appeler AVANT un gros entraînement.

    Sur vast.ai tu paies à l'heure : si le GPU n'est pas vu, tu entraînerais
    sur CPU en payant l'instance. Cette fonction crie fort dans ce cas.
    """
    if torch.cuda.is_available():
        print("[check] GPU CUDA détecté, entraînement à pleine vitesse.")
    elif torch.backends.mps.is_available():
        print("[check] MPS détecté (local). OK pour petits modèles / smoke test.")
    else:
        print("=" * 60)
        print("ATTENTION : aucun GPU détecté.")
        print("Sur une instance louée, tu paierais pour de l'entraînement CPU.")
        print("Vérifie l'image Docker (torch+CUDA) avant de continuer.")
        print("=" * 60)


if __name__ == "__main__":
    # Petit test : affiche ce que la machine courante propose.
    dev = get_device()
    print(f"Device retenu : {dev}")
    assert_gpu_or_warn()
