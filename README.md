# Détection sémantique de changements — Projet Fil Rouge Airbus

Détection de changement sémantique urbain sur le dataset **Hi-UCD** (Tallinn,
imagerie aérienne 0,1 m, 2 dates). Réseau siamois middle-fusion, full supervisé.

## Structure du projet

```
projet-airbus-scd/
├── notebooks/        # exploration : masque 3 canaux, distributions, visus
├── src/
│   ├── data/         # dataset Hi-UCD, lecture du masque, dataloader
│   ├── models/       # encodeur siamois, têtes, fusion
│   ├── losses/       # CE / Dice / Focal + pondération multi-tâche
│   ├── utils/        # device (MPS/CUDA/CPU), helpers
│   ├── train.py      # boucle d'entraînement
│   └── evaluate.py   # mIoU / Sek / Fscd / BCD IoU + seuil
├── configs/          # 1 expérience = 1 fichier YAML
├── scripts/          # utilitaires one-shot (préparation données, etc.)
└── outputs/          # checkpoints, logs, figures (git-ignorés)
```

**Règle de partage notebook / scripts :**
- *notebook* = ce qui se regarde une fois (explorer, visualiser, comprendre)
- *script*   = ce qui se rejoue, se teste, s'importe (data, modèle, train, eval)

**Règle des expériences :** une expérience est un fichier de config, pas une
copie de code. `python -m src.train --config configs/exp5.yaml`.

## Workflow en deux environnements

Le code est écrit UNE fois et tourne partout grâce à `src/utils/device.py`
(auto-détection MPS / CUDA / CPU).

### En local (Mac, GPU Apple Silicon via MPS)
Pour : exploration, développement, débogage, **smoke tests**.

```bash
source env_fil_rouge/bin/activate
pip install -r requirements.txt      # n'installe PAS torch (déjà présent)
python -m src.utils.device           # vérifie que MPS est vu
```

### Sur vast.ai (GPU NVIDIA via CUDA)
Pour : les gros entraînements uniquement.

1. Lancer une instance avec une **image Docker PyTorch + CUDA** (torch déjà
   câblé au GPU — ne PAS le réinstaller).
2. Sur l'instance :
   ```bash
   git clone <ton-repo> && cd projet-airbus-scd
   pip install -r requirements.txt   # métier seulement
   python -m src.utils.device        # DOIT afficher "CUDA actif"
   ```
3. **Garde-fou anti-gaspillage** : avant tout entraînement long, vérifie que
   `torch.cuda.is_available()` est `True`. Sinon tu paies l'instance pour de
   l'entraînement CPU. Le module `device.py` le contrôle automatiquement.

### Pourquoi torch n'est pas dans requirements.txt
La build torch diffère par machine : MPS en local, CUDA sur vast.ai. Figer
une version unique casserait le lien au GPU d'un côté ou de l'autre. Le code
"métier" (numpy, dataloader, modèle…) est identique partout ; torch ne l'est
pas. C'est l'erreur la plus courante sur GPU loué — évitée ici par conception.

## Économie de crédits
Tout ce qui n'a pas besoin de puissance se fait gratuitement en local :
écrire et débugger le code, faire tourner 1 epoch sur 10 tuiles (smoke test),
tracer les courbes, choisir le seuil. vast.ai ne sert qu'à mettre de la
puissance sur du code déjà validé en local.

## Données
Hi-UCD complet, 2 dates. Masque = PNG 3 canaux dans `mask/` :
- canal 1 : classes occupation du sol T1 (0–9)
- canal 2 : classes occupation du sol T2 (0–9)
- canal 3 : changement (0=unlabeled, 1=no-change, 2=change)

La classe `unlabeled` (canal 3 = 0) est exclue de la loss et des métriques.
Les données ne sont pas dans le repo (cf. `.gitignore`).
```
```
