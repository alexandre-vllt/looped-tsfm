# Izar — journal de bord et plan

Projet *Looped Time Series Foundation Models* · EPFL, LIONS lab
Cluster SCITAS Izar, compte `agvallet`, account Slurm `master`
Dernière mise à jour : 29.09.2026

> **Échéance à noter** : le corpus synthétique a été généré le 28.09 sur
> `/scratch`, qui purge à 30 jours. **Il disparaît autour du 28.10** si le
> `/work` n'est pas obtenu d'ici là. Le régénérer coûte 52 minutes, donc ce
> n'est pas dramatique — mais il faut y penser.

---

## 1. Accès et environnement

| | |
|---|---|
| Connexion | `ssh izar` (alias dans `~/.ssh/config`, clé ed25519) |
| Login node | `izar1` — **pas de GPU** |
| Partitions | `gpu` (volta:2 / volta:4), `gpu-xl` (volta:4), `test` (ampere:2) |
| QOS | `build`, `debug`, `long`, `normal` |
| GPU | Tesla V100-PCIE-32GB, compute capability **7.0 (sm_70)** |
| Coût | CHF 0.0000 / GPU / heure, jusqu'à 16 GPU |
| `/home` | 100 GB, sauvegardé — le code |
| `/scratch/izar/agvallet` | **purge à 30 jours** — les données |
| `/work` | pas obtenu, à demander à Arber |
| Expiration du compte | février 2027 |

Le seul nœud Ampere (`i70`) est en `drain*` — `#Node removed and put into
Manticore`. Izar est exclusivement Volta.

```bash
# ~/env.sh
module purge
module load gcc/11.3.0 cuda/12.1.1
export PATH="$HOME/.local/bin:$PATH"
export TRITON_CACHE_DIR=$HOME/.triton_cache
export PYTHONPATH=$HOME/sem_proj/reverso:$PYTHONPATH
export GIFT_EVAL=/scratch/izar/agvallet/gift-eval
source ~/venvs/looped311/bin/activate
```

Python **3.11** (via `uv`, SCITAS n'a que 3.10.4), torch **2.7.1+cu126**,
triton 3.3.1, flash-linear-attention 0.5.2, gluonts 0.15.x.
FlashFFTConv **non installé** — ne cible que A100/H100.

---

## 2. Ce qui est fait

### 2.1 Environnement — trois murs franchis

| # | Symptôme | Vraie cause | Correction |
|---|---|---|---|
| 1 | `Autotuner.__init__() got an unexpected keyword 'do_bench'` | triton 3.1 trop ancien pour `fla` | torch → 2.6.0 |
| 2 | `AttributeError: 'NoneType' ... .start()` dans `jit.py` | **Python 3.10** : `inspect.getblock` coupe le source sur les décorateurs `@triton.autotune` multi-lignes | Python 3.11 via `uv` |
| 3 | `assert q.dtype != torch.float32` | `chunk_delta_rule` n'accepte pas le fp32 | bf16 (ce que font les auteurs) |

### 2.2 Verdict matériel

Les kernels **chunkés de `fla` ne compilent pas sur sm_70**, en fp16 comme en
bf16 — échec du pass `ConvertTritonGPUToLLVM`. Ce n'est pas réparable côté
utilisateur. Vérifications qui écartent les autres hypothèses : `sm_70` est
bien dans `torch.cuda.get_arch_list()`, et le matmul bf16 fonctionne.

Reste `mode='fused_recurrent'` : mathématiquement identique, sérialisé sur les
2048 pas au lieu d'être chunké.

### 2.3 Les trois implémentations, et l'hybride

| | convolution | DeltaNet | V100 |
|---|---|---|---|
| `reverso/` | FlashFFTConv | `fla` chunk | non |
| `reverso_torch/` | `torch.fft` | boucle Python | oui, très lent |
| **`reverso_fast/`** | `torch.fft` | `fla` fused_recurrent | **oui** |

Validation à trois branches (`cmp3.py`), pour séparer l'effet du dtype de
celui du kernel :

```
sine   torch fp32 vs torch bf16    corr 0.999959
       torch fp32 vs fla   bf16    corr 0.999942   <- le kernel n'ajoute rien
```

### 2.4 Performance mesurée

Reverso-Small complet, forward + backward :

| B | ms/step | tok/s | mémoire |
|---|---|---|---|
| 32 | 221.6 | 296k | 0.78 GB |
| 64 | 422.0 | 311k | 1.54 GB |
| 128 | 823.9 | **318k** | 3.06 GB |

Saturé dès B=64, 3 GB utilisés sur 32. La limite est la sérialisation
temporelle, pas le matériel. **Le batch size n'influence donc plus le débit**
et se choisit selon la recette, pas selon la machine.

À 318k tok/s et 1.05M points par step (batch 512 × 2048, le chiffre du
papier), **un step coûte 3.3 s sur une V100**, ou ~0.41 s sur 8 en DDP.

### 2.5 L'étalon GIFT-Eval

Reproduction du chiffre publié avec le checkpoint officiel, à travers notre
chaîne complète :

```
97 taches
geo mean MASE modele  = 1.0259
geo mean MASE naive   = 1.3979
SCORE GIFT-Eval       = 0.7339      publie 0.726
```

**1.1 % d'écart.** La chaîne est validée : poids, architecture, hybride,
normalisation min-max, rollout autorégressif, métriques.

Le résidu vient d'un Seasonal Naive recalculé plutôt que celui exact des
auteurs, et de `--force-flip-invariance` non activé.

Point de méthode : le chiffre du leaderboard est la moyenne géométrique du
MASE **relatif au Seasonal Naive**, pas du MASE brut. Le MASE brut tourne
autour de 1 puisqu'il est déjà divisé par l'erreur naïve in-sample.

### 2.6 Le papier, lu et dépouillé

| | |
|---|---|
| §3.1 | AdamW lr 5e-4, β (0.9, 0.999), ε 1e-8, wd 0.1, WSD, batch 512, ~1M points/step, {10, 20, 40} H100-h pour Nano/Small/Reverso |
| §3.1 objectif | **MAE sur la sortie dénormalisée**, NaN masqués — j'avais initialement mis un MSE normalisé, corrigé |
| §2.1 | normalisation **min-max vers [0,1]**, multi-scale à C = 4 canaux (contexte effectif 16384) |
| Table 9 | banque de noyaux : Constant, Linear, RBF, Rational Quadratic, **Matérn**, Periodic |
| Algorithmes 1–4 | KernelSynth, spike process, TSI, pipeline d'augmentation |
| Table 14 | **confirme que les checkpoints publiés sont les variantes « w/o multi-scale »** : 0.760 / 0.726 / 0.711, exactement les chiffres du README |
| Table 10 | MASE par dataset pour le modèle 2.6M — utilisable comme validation fine |

### 2.7 Le corpus synthétique

Fidèle aux Algorithmes 1–3 et à la Table 9. **1 000 000 séries de 2096 points,
générées en 51.5 minutes à 323.7 séries/s, 7.9 GB sur `/scratch`.**

Deux détails d'implémentation qui ont compté. `torch.linalg.cholesky` lève si
**une seule** matrice du batch échoue, ce qui relançait les 16 à chaque fois —
remplacé par `cholesky_ex`, qui signale les échecs par matrice. Et un premier
essai naïf matérialisait deux tenseurs de 281 MB par tentative, saturant
l'allocateur CUDA.

**Un point de fidélité à signaler** : près de 14 matrices sur 16 échouent à la
Cholesky au premier jitter, et il faut deux escalades — soit un jitter de
**1e-2 relatif à la diagonale**, c'est-à-dire du bruit blanc à 1 % du signal
ajouté à chaque série. Ça vient de la banque elle-même (`Constant` est de rang
1, `Linear` avec σ=0 aussi, et les produits empilent les dégénérescences). Le
papier mentionne la Cholesky batchée sans jamais parler de régularisation.
**À demander aux auteurs.**

### 2.8 Le code écrit

```
~/sem_proj/reverso/
  reverso_fast/           l'hybride (copie de reverso_torch + fla)
  synthetic.py            Algorithmes 1-3 + Table 9
  gen_synthetic.py        generation offline en shards
  synth_dataset.py        loader memory-mappe
  augment.py              Algorithme 4 (pour les donnees reelles)
  plot_synth.py           inspection visuelle + statistiques du corpus
  profile_synth.py        ou passe le temps de generation
  train.py                boucle d'entrainement, hyperparametres du papier
  cmp3.py                 test d'equivalence a trois branches
  bench_bs.py             balayage de batch size
  loop_norms.py           croissance de norme sous looping (non lance)
  example/eval_gift.py    patche : --fast, --naive
  results/reverso_small_full/   MASE du modele, 97 taches
  results/seasonal_naive/       MASE de la baseline, 97 taches
/scratch/izar/agvallet/
  synth/                  1M series, 50 shards, 7.9 GB
  gift-eval/              donnees d'evaluation
```

Le patch de looping (`loop_k`, `loop_inject`, `res_scale = 1/k`) est écrit
mais **non appliqué** — décision de finir la baseline d'abord.

---

## 3. Ce qui reste, et qui est faisable sur Izar

Par ordre de dépendance. Les trois premiers groupes ne demandent aucune
réponse extérieure.

### A. Achever la validation — quelques heures

**A1. Inspecter le corpus synthétique.** `plot_synth.py`, déjà écrit. Trois
critères : part de séries constantes proche de 0, aucune série non finie, et
une fraction significative avec un pic spectral marqué. Sans ce dernier point,
le modèle n'apprendra jamais de saisonnalité.

**A2. Vérifier les gradients.** `cmp3.py` ne teste que le forward. Sur le
backward, on a seulement vérifié que ça ne plante pas, jamais que c'est juste.
Pour un projet dont tout dépend de l'entraînement, c'est un trou. Une trentaine
de lignes : comparer les gradients entre `reverso_torch` et `reverso_fast` sur
une petite entrée.

**A3. Valider finement contre la Table 10.** Évaluer `checkpoints/reverso/`
(2.6M) et comparer **dataset par dataset** aux valeurs du papier, plutôt que
sur le seul agrégat. Beaucoup plus discriminant : si une seule tâche diverge,
on le voit.

### B. La baseline — quelques jours

**B4. Premier vrai run sur le corpus synthétique.** Reverso-Small, batch 512,
les hyperparamètres du papier. Le critère de succès n'est pas une loss qui
descend, c'est **battre nettement les trois baselines naïves** affichées au
démarrage.

**B5. DDP multi-GPU.** Jusqu'à 16 GPU, soit ~0.2 s/step. C'est ce qui rend les
runs longs praticables.

**B6. Chaînage de jobs.** Aucun job ne tourne plusieurs jours : il faut
`--dependency=afterany` et une reprise exacte. La reprise est déjà écrite
(poids, optimizer, scheduler, RNG, position du dataset) ; il manque le
script d'enchaînement.

**B7. Évaluer le modèle entraîné sur GIFT-Eval.** Premier chiffre à soi,
comparable à l'étalon 0.7339. Il sera mauvais — pas de données réelles — mais
il devient la référence pour tous les runs bouclés.

### C. Le looping — le cœur du sujet

**C8. Appliquer le patch de looping.** `loop_k`, `loop_inject`,
`res_scale = 1/k`.

**C9. Non-régression.** À `k=1`, `cmp3.py` doit redonner exactement les mêmes
chiffres qu'avant le patch. C'est la garantie que le code de looping n'a pas
changé la sémantique du modèle de base.

**C10. Mesurer la croissance de norme.** `loop_norms.py`, déjà écrit. Compare
`res_scale = 1` et `res_scale = 1/k` pour k ∈ {1,2,4,8}. Donne une figure pour
le rapport et justifie empiriquement le choix.

**C11. Looping sans entraînement.** Le papier *Training-free Looped
Transformers* du second dossier propose de boucler un modèle déjà entraîné.
Si ça s'applique, tu obtiens une **courbe MASE-vs-k en quelques heures de GPU,
sur les checkpoints publiés, sans entraîner quoi que ce soit**. C'est le
meilleur rapport temps/valeur disponible, et un résultat préliminaire à
montrer à Arber. À lire en priorité.

**C12. Balayage k sur Reverso-Nano.** 190K paramètres, une seule couche
DeltaNet. k ∈ {1,2,4} tient sur Izar et donne déjà un signal.

### D. La question d'expressivité — la plus intéressante

**D13. Tâches de state-tracking synthétiques.** Parité, arithmétique modulaire,
composition dans A₅. C'est le test direct du « looping linear RNNs can lead to
nonlinearities (to be checked) » d'Arber. Données générées à la volée, modèles
minuscules, aucun stockage, quelques heures de V100. **Aucune dépendance
extérieure.**

**D14. Ablation `allow_neg_eigval`.** Le bouton est dans le code, désactivé,
et l'activer ne coûte ni paramètre ni calcul. La grille `k × allow_neg_eigval`
croise directement ta question : looping et valeurs propres négatives
achètent-ils la même expressivité, ou sont-ils complémentaires ?

### E. Les données réelles — le gros morceau

**E15. Corpus GiftEval pretrain.** 4.5M séries, 230 milliards de points. Le
papier décrit le schéma d'échantillonnage : N_max = 100 000 échantillons par
dataset et par epoch, stride recalculé par dataset, plafond de 48 échantillons
par série, point de départ aléatoire à chaque epoch. Reste à écrire le
`Dataset` et à y brancher `augment.py`.

**Le vrai obstacle ici est le stockage**, pas le code : `/scratch` purge à 30
jours et `/home` fait 100 GB. C'est ce qui rend la demande de `/work` à Arber
plus urgente que celle des GPU.

---

## 4. Ce qui n'est pas faisable sur Izar

- **Le balayage complet** k ∈ {1,2,4,8} × 3 seeds sur Small ou 2.6M : environ
  15 runs-équivalents, soit ~165 h sur 8 GPU. Faisable en théorie, déraisonnable
  en pratique avec les files d'attente.
- **Tout ce qui demande les kernels chunkés** pour la vitesse.
- **Une reproduction exacte du papier**, tant que manquent : fractions du WSD,
  nombre de steps, ratio réel:synthétique, EMA, gradient clipping, probabilités
  d'augmentation, plages du spike process, probabilités TSI, et la
  régularisation de la Cholesky.

---

## 5. Pièges rencontrés, à ne pas refaire

**Tout ce qui importe `fla` doit passer par `srun`.** Le nœud de login n'a pas
de driver CUDA. Pour vérifier qu'un module existe sans l'exécuter :
`python -c "import importlib.util as u; print(u.find_spec('x').origin)"`.

**L'autotuner Triton relance un sweep complet à chaque nouvelle forme de
tenseur** (~18–21 s). Batch fixe pendant tout un run, et `TRITON_CACHE_DIR`
persistant entre jobs.

**Un `pip install -e .` fige la liste des paquets.** Un module créé après
l'install n'est pas importable. Réglé par `PYTHONPATH`.

**`sbatch` écrit le `.out` dans le répertoire de soumission**, pas celui du
script.

**Vérifier la longueur après un `merge` pandas.** Une jointure silencieusement
incomplète a produit un score de 0.6129 au lieu de 0.734 — « meilleur que
publié », ce qui était le signal d'alarme. Toujours `assert len(df) == len(m)`.

**`scp` se lance depuis la machine locale**, jamais depuis le cluster. Izar ne
peut pas initier de connexion sortante.

**`huggingface-cli` est déprécié et ne télécharge rien**, en silence. Utiliser
`hf download`.

**Le cache DNS de macOS.** `Could not resolve hostname` avec le VPN actif :
`sudo dscacheutil -flushcache && sudo killall -HUP mDNSResponder`, ou passer par
l'IP dans `~/.ssh/config`.

**Pas de `tmux` sur izar.** `sbatch` (détaché par construction) ou `nohup`.

**Toujours imprimer une ligne de progression.** Un job qui n'affiche rien
pendant vingt minutes est indiscernable d'un job bloqué.

---

## 6. En attente de l'extérieur

| | qui | pour quoi |
|---|---|---|
| 2 nœuds A100 | Arber, annoncé | débloque les kernels chunkés, donc le balayage complet |
| Kuma | Arber, demandé | idem, à plus grande échelle |
| `/work` | à demander | `/scratch` purge à 30 jours, le projet dure 4 mois |
| Code d'entraînement | email à Xinghong Fu | éliminerait d'un coup toutes les inconnues ci-dessous |
| Hyperparamètres manquants | même email | WSD, steps, ratio, EMA, clipping, probabilités d'augmentation, plages spike/TSI, régularisation Cholesky |
