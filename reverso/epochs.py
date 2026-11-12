"""Enumeration par epoch, pour la pipeline v2.

POURQUOI CE MODULE EXISTE
`DeterministicMix` tire AVEC REMISE : l'echantillon n est une fonction pure de
(n, seed), ce qui donne la reproductibilite, mais sur 150 000 steps x 512
certaines series sont vues des dizaines de fois et d'autres jamais. Le mot
"epoch" n'a alors aucun sens, on ne peut pas parler de surapprentissage, et on
ne peut pas definir une duree d'entrainement autrement qu'en la fixant a la
main.

`EnumeratedMix` garde la purete -- l'echantillon n reste une fonction pure de
(n, seed) -- et ajoute la couverture EXACTE : a chaque epoch, chaque chunk
alloue est vu une fois et une seule, sur une fenetre differente.

COMMENT LA PURETE SURVIT A L'ENUMERATION
L'epoch est encodee DANS l'ordinal : epoch = ordinal // E, position dans
l'epoch = ordinal % E. Il n'y a donc aucun etat a avancer, aucun `set_epoch`
a propager aux workers, et `start_step` reprend au milieu exactement comme
avant. Le DataLoader reste un SequentialSampler sur les ordinaux.

L'ORDRE VIENT D'UNE PERMUTATION FEISTEL, PAS D'UN TABLEAU
Melanger 500 000 indices demanderait de stocker la permutation et de la
partager entre workers. Une permutation preservant le format (un reseau de
Feistel sur quelques bits, avec cycle-walking pour les domaines qui ne sont
pas une puissance de deux) donne la meme chose comme FONCTION : perm(i) se
calcule en quatre tours d'arithmetique entiere, sans memoire, et c'est une
bijection sur [0, n). Cle = (seed, epoch), donc l'ordre change a chaque epoch
et reste reproductible.

LA TENSION AVEC LA PONDERATION alpha -- A LIRE AVANT DE CHOISIR
v1 ponderait chaque source par n^0.5 (done.md 5.3.1) pour que les grosses
sources n'ecrasent pas les petites. Enumerer exactement et ponderer sont
INCOMPATIBLES : on ne peut pas voir chaque chunk une fois et en meme temps
sur-echantillonner les petites sources. D'ou les deux politiques :

  policy="enumerate"  chaque chunk une fois par epoch. alpha vaut 1.0 de
                      fait : une source pese exactement sa taille. C'est la
                      lecture litterale de "entrainer sur tout le train set".
                      Rien n'est sur-echantillonne, donc la pathologie de
                      alpha=0 (une petite source tiree 15 686 fois) n'existe
                      pas ici -- le risque est seulement que les grosses
                      sources dominent la diversite de domaine.
  policy="alpha"      la source g recoit round(n_g^alpha / Z * S) creneaux par
                      epoch. On garde la decision alpha=0.5 mesuree en v1,
                      mais "exactement une fois" devient "exactement une fois
                      par creneau alloue" : les petites sources se repetent a
                      l'interieur d'une epoch, les grosses sont sous-
                      echantillonnees et couvertes sur plusieurs epochs grace
                      au decalage (r + e*slots) mod n.

Les deux donnent des epochs de MEME taille, donc le meme nombre de steps : la
politique change ce qu'on voit, pas combien.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
from numpy.random import Generator, Philox
from torch.utils.data import Dataset

_M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15

# Separe les flux : la cle d'offset de fenetre ne doit pas coincider avec la
# cle de permutation, sinon l'offset serait correle a la position.
_KEY_PERM_REAL = 0x5BF03635EC2A9C19
_KEY_PERM_SYN = 0xD6E8FEB86659FD93
_KEY_SLOT = 0xA0761D6478BD642F
_KEY_OFFSET = 0x8EBC6AF09C88C6E3


def _mix64(z: int) -> int:
    """splitmix64. Bon melange, quatre operations, pas d'etat."""
    z = (z + _GOLDEN) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return (z ^ (z >> 31)) & _M64


class Feistel:
    """Permutation preservant le format sur [0, n), sans memoire.

    Reseau de Feistel equilibre sur `bits` bits (bits pair, 2**bits >= n),
    puis cycle-walking : on itere la permutation jusqu'a retomber dans
    [0, n). C'est une bijection sur [0, n) parce que la permutation sur
    [0, 2**bits) est une bijection et qu'on suit son cycle. Comme
    2**bits < 4n, l'esperance du nombre d'iterations est inferieure a 4.
    """

    __slots__ = ("n", "half", "mask", "key", "rounds")

    def __init__(self, n: int, key: int, rounds: int = 4):
        self.n = int(n)
        if self.n <= 0:
            raise ValueError(f"Feistel: n doit etre > 0, recu {n}")
        bits = max(2, (self.n - 1).bit_length())
        if bits % 2:
            bits += 1
        self.half = bits // 2
        self.mask = (1 << self.half) - 1
        self.key = int(key) & _M64
        self.rounds = int(rounds)

    def _round(self, x: int) -> int:
        lo = x & self.mask
        hi = (x >> self.half) & self.mask
        for r in range(self.rounds):
            lo, hi = hi, (lo ^ (_mix64(self.key ^ (hi << 8) ^ r) & self.mask))
        return ((hi << self.half) | lo) & ((1 << (2 * self.half)) - 1)

    def __call__(self, i: int) -> int:
        if not 0 <= i < self.n:
            raise IndexError(f"Feistel: {i} hors de [0, {self.n})")
        if self.n == 1:
            return 0
        x = i
        for _ in range(64):            # garde-fou : jamais atteint en pratique
            x = self._round(x)
            if x < self.n:
                return x
        raise RuntimeError("Feistel: cycle-walking n'a pas converge")


def allocate(counts: List[int], total: int, alpha: Optional[float]) -> List[int]:
    """Creneaux par source dans une epoch.

    alpha is None  -> enumeration pure : slots_g = counts_g (somme = total).
    alpha = a       -> slots_g proportionnel a counts_g**a, somme = total,
                       chaque source gardant au moins un creneau.

    Le reste de l'arrondi va aux sources dont la part fractionnaire est la
    plus grande (methode du plus fort reste), pour que la somme soit exacte.
    """
    if alpha is None:
        return [int(c) for c in counts]

    w = np.asarray(counts, dtype=np.float64) ** float(alpha)
    exact = w / w.sum() * float(total)
    base = np.maximum(1, np.floor(exact).astype(np.int64))
    short = int(total) - int(base.sum())
    if short > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for i in range(short):
            base[order[i % len(base)]] += 1
    elif short < 0:
        order = np.argsort(exact - np.floor(exact))
        i = 0
        while short < 0 and i < 10 * len(base):
            g = order[i % len(order)]
            if base[g] > 1:
                base[g] -= 1
                short += 1
            i += 1
    return [int(x) for x in base]


class _Side:
    """Un corpus avec son allocation par epoch et ses permutations."""

    def __init__(self, corpus, total: int, alpha: Optional[float],
                 seed: int, perm_key: int):
        self.c = corpus
        self.alpha = alpha
        self.slots = allocate(list(corpus.counts), int(total), alpha)
        self.total = int(sum(self.slots))
        self.cum = np.cumsum(self.slots)
        self.seed = int(seed)
        self.perm_key = int(perm_key)
        self._cache: Tuple[int, Optional[Feistel], list] = (-1, None, [])

    def _perms(self, epoch: int):
        """Permutations de l'epoch, mises en cache : une epoch = des milliers
        d'appels, et construire un Feistel est quelques operations mais
        inutiles a repeter."""
        e, outer, inner = self._cache
        if e == epoch:
            return outer, inner
        k = _mix64(self.perm_key ^ (self.seed << 1) ^ (epoch * _GOLDEN))
        outer = Feistel(self.total, k)
        inner = [Feistel(int(n), _mix64(k ^ (g + 1) * _GOLDEN))
                 for g, n in enumerate(self.c.counts)]
        self._cache = (epoch, outer, inner)
        return outer, inner

    def locate(self, epoch: int, slot: int) -> Tuple[int, int]:
        """Creneau -> (fichier, ligne). Bijectif sur les creneaux de l'epoch."""
        outer, inner = self._perms(epoch)
        s = outer(slot)                              # melange les sources
        g = int(np.searchsorted(self.cum, s, side="right"))
        g = min(g, len(self.slots) - 1)
        r = s - (int(self.cum[g - 1]) if g else 0)   # rang dans la source
        n = int(self.c.counts[g])
        # Decalage par epoch : une source sous-echantillonnee (slots < n) est
        # couverte sur plusieurs epochs au lieu de revoir les memes lignes.
        return g, inner[g]((r + epoch * self.slots[g]) % n)


class EnumeratedMix(Dataset):
    """Ordinal -> (contexte, cible), avec de vraies epochs.

    epoch       = ordinal // E
    position    = ordinal %  E
    E           = taille d'une epoch en echantillons

    Les creneaux reels et synthetiques sont entrelaces par Bresenham : le
    creneau j est reel si floor((j+1)*R/E) > floor(j*R/E). Le compte est donc
    EXACT (R reels par epoch, pas "en esperance") et regulierement reparti,
    sans paquets de reels suivis de paquets de synthetiques.
    """

    def __init__(
        self,
        synthetic,
        real=None,
        seq_len: int = 2048,
        horizon: int = 48,
        seed: int = 0,
        p_real: float = 0.5,
        alpha: Optional[float] = 0.5,
        max_epochs: int = 64,
    ):
        if real is None or p_real <= 0.0:
            raise ValueError("EnumeratedMix veut un corpus reel et p_real > 0 : "
                             "l'epoch est definie par un passage sur le reel")
        self.seq_len, self.horizon = int(seq_len), int(horizon)
        self.span = self.seq_len + self.horizon
        self.seed = int(seed)
        self.p_real = float(p_real)
        self.alpha = alpha
        self.max_epochs = int(max_epochs)

        n_real = int(sum(real.counts))
        self.R = n_real                                   # creneaux reels
        self.E = int(round(n_real / self.p_real))         # taille d'epoch
        self.S = self.E - self.R                          # creneaux synthetiques
        if self.S <= 0:
            raise ValueError(f"p_real {p_real} ne laisse pas de place au "
                             f"synthetique (E={self.E}, R={self.R})")

        self.real = _Side(real, self.R, alpha, self.seed, _KEY_PERM_REAL)
        self.syn = _Side(synthetic, self.S, alpha, self.seed, _KEY_PERM_SYN)
        self.return_ordinal = False

    # -- geometrie de l'epoch ----------------------------------------------

    def steps_per_epoch(self, batch_size: int) -> int:
        return self.E // int(batch_size)

    def describe(self, batch_size: int = 512) -> str:
        pol = "enumeration pure (alpha=1)" if self.alpha is None \
            else f"ponderee alpha={self.alpha:g}"
        big = max(self.real.slots)
        return (f"EnumeratedMix : epoch = {self.E:,} fenetres "
                f"({self.R:,} reelles + {self.S:,} synthetiques), "
                f"{self.steps_per_epoch(batch_size):,} steps a batch "
                f"{batch_size}\n"
                f"  politique sources : {pol}\n"
                f"  reel : {len(self.real.slots)} sources, plus grosse part "
                f"{big / self.R * 100:.1f} % de l'epoch\n"
                f"  max_epochs {self.max_epochs} -> "
                f"{self.E * self.max_epochs:,} ordinaux adressables")

    def __len__(self) -> int:
        return self.E * self.max_epochs

    # -- la carte pure ------------------------------------------------------

    def __getitem__(self, ordinal: int):
        o = int(ordinal)
        epoch, j = divmod(o, self.E)

        # Bresenham : compte exact, repartition reguliere, pas de hasard.
        r_before = (j * self.R) // self.E
        r_after = ((j + 1) * self.R) // self.E
        use_real = r_after > r_before

        if use_real:
            side, slot = self.real, r_before
        else:
            side, slot = self.syn, j - r_before

        g, row = side.locate(epoch, slot)

        arr = side.c.arrays[g]
        L = int(side.c.lengths[g])
        slack = L - self.span
        if slack < 0:
            raise ValueError(f"{side.c.label}: {side.c.names[g]} fait {L} "
                             f"points, un echantillon en demande {self.span}")
        # Offset tire d'un flux propre, fonction pure de (epoch, fichier,
        # ligne) : la meme ligne revue a l'epoch suivante donne une AUTRE
        # fenetre, ce qui est tout l'interet de l'enumeration par epoch.
        rng = Generator(Philox(key=self.seed ^ _KEY_OFFSET,
                               counter=(epoch << 44) ^ (g << 36) ^ row))
        off = int(rng.integers(slack + 1)) if slack else 0

        seq = np.array(arr[row, off:off + self.span], dtype=np.float32)
        x = torch.from_numpy(seq[: self.seq_len]).unsqueeze(-1)
        y = torch.from_numpy(seq[self.seq_len:]).unsqueeze(-1)
        return (x, y, o) if self.return_ordinal else (x, y)
