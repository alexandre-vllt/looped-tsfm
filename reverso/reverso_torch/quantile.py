"""Tete de regression quantile sur un backbone gele.

POURQUOI
Notre modele sort une prevision PONCTUELLE de 48 points, entrainee par MAE
(reverso_loss, papier 2.1). Mais GIFT-Eval note aussi des metriques
probabilistes -- CRPS et mean_weighted_sum_quantile_loss sont dans la liste
METRICS de example/eval_gift.py. Et voila comment on les obtient aujourd'hui :

    samples = np.repeat(pred_full_np[i][None, :], self.num_samples, axis=0)

Cent copies identiques du point. Tous les quantiles egaux, la distribution
predite est un Dirac, le CRPS degenere. La moitie probabiliste du benchmark
n'est pas jouee.

LA RECETTE EST CELLE DU PAPIER, C.5 "Probabilistic Forecasting" :
  "we repurpose a pretrained Reverso model by training the decoder head with a
   quantile regression objective [...] does not train the full model from
   scratch"
Backbone GELE. Leur Table 7 : CRPS 0.459 sur le split medium, meilleur score
du tableau toutes tailles confondues.

QUELLE COUCHE REMPLACER -- lu dans model.py lignes 376-405, pas suppose
    self.head     = nn.Linear(input_token_len, output_bottleneck_dim)   2048 -> 48
    self.out_proj = nn.Linear(d_model, 1)                                 32 -> 1

    temp_out = self.head(dec_out).permute(0, 2, 1)
    q = self.simple_q_proj(temp_out)      # les 48 REQUETES du decodeur
    k = self.key_proj(dec_out_perm)
    v = self.value_proj(dec_out_perm)
    ...
    dec_out = self.out_proj(attn)         # <- LA SORTIE

`head` fabrique les 48 positions de requete a partir du contexte ; ce n'est
PAS la sortie, malgre son nom et malgre ses 48 canaux. La couche de sortie est
`out_proj`, qui projette chaque position attendue (d_model = 32) vers un
scalaire. La premiere version de ce module visait `head` -- cherche "le Linear
dont out_features = 48" -- ce qui aurait entraine une tete sur les requetes du
decodeur. Nombres plausibles, resultat faux.

Les formes tombent juste : `attn` est (B, 48, 32), donc Linear(32, Q) donne
directement (B, 48, Q).

LE WARM START
La tete quantile recopie Q fois les poids de `out_proj`. A l'etape 0 tous les
quantiles valent donc exactement la prevision ponctuelle, et la perte pinball
vaut EXACTEMENT la moitie de la MAE du modele de depart (a q=0.5 la pinball
vaut 0.5|e|, et les autres niveaux se compensent deux a deux sur une erreur
symetrique). C'est un controle verifiable : si les deux ne coincident pas a
~1 %, le hook n'est pas sur la bonne couche et il ne faut PAS entrainer.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Les neuf quantiles de MeanWeightedSumQuantileLoss du harnais. Ne pas changer
# sans changer la metrique : le CRPS rapporte est une somme ponderee SUR CES
# niveaux-la.
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# model.py 380 : self.out_proj = nn.Linear(self.d_model, 1)
DEFAULT_HEAD = "out_proj"


def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 quantiles: Tuple[float, ...],
                 scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Perte quantile (pinball).

    pred    (B, H, Q)  une prediction par niveau de quantile
    target  (B, H)     la cible
    scale   (B,)       diviseur par fenetre, pour une perte sans echelle.
                       None = perte brute.

    Pour le niveau q :  L_q = max( q(y - yhat), (q-1)(y - yhat) )
    soit q(y - yhat) quand on sous-estime, (1-q)(yhat - y) quand on surestime.

    A q = 0.5 cela vaut 0.5 |y - yhat|, la MOITIE de la MAE -- pas la MAE. Le
    facteur 1/2 est intrinseque a la definition ; ne pas le "corriger", sinon
    la mediane n'est plus comparable aux autres niveaux.
    """
    if pred.dim() != 3:
        raise ValueError(f"pred doit etre (B, H, Q), recu {tuple(pred.shape)}")
    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    if pred.shape[-1] != q.numel():
        raise ValueError(f"pred a {pred.shape[-1]} quantiles, {q.numel()} attendus")

    err = target.unsqueeze(-1) - pred                      # (B, H, Q)
    loss = torch.maximum(q * err, (q - 1.0) * err)
    if scale is not None:
        loss = loss / scale.view(-1, 1, 1).clamp_min(1e-8)
    return loss.mean()


def find_head(model: nn.Module, name: str = DEFAULT_HEAD) -> Tuple[str, nn.Linear]:
    """La couche de sortie, par NOM et non par heuristique.

    On ne cherche plus "le Linear dont out_features vaut 48" : c'est ce qui
    designait `head`, qui produit les requetes du decodeur. Le nom par defaut
    vient de model.py ligne 380.
    """
    mods = dict(model.named_modules())
    if name not in mods:
        cands = [n for n, m in mods.items() if isinstance(m, nn.Linear)]
        raise KeyError(f"{name!r} absent. Linear disponibles : {cands}")
    m = mods[name]
    if not isinstance(m, nn.Linear):
        raise TypeError(f"{name!r} est un {type(m).__name__}, pas un nn.Linear")
    if m.out_features != 1:
        raise ValueError(
            f"{name!r} sort {m.out_features} canaux, 1 attendu. La couche de "
            f"sortie de Reverso est out_proj: Linear(d_model, 1) -- une valeur "
            f"par position. Un Linear a 48 sorties est `head`, qui fabrique les "
            f"requetes du decodeur, PAS la prevision.")
    return name, m


class QuantileHead(nn.Module):
    """Q copies de la couche de sortie, entrainees separement.

    Initialisees IDENTIQUES a l'originale, donc a l'etape 0 la sortie est
    exactement la prevision ponctuelle repetee Q fois -- le meme Dirac
    qu'aujourd'hui, mais comme point de depart d'un entrainement plutot que
    comme resultat final.
    """

    def __init__(self, src: nn.Linear, n_quantiles: int):
        super().__init__()
        if src.out_features != 1:
            raise ValueError(f"src doit sortir 1 canal, pas {src.out_features}")
        self.q = int(n_quantiles)
        self.lin = nn.Linear(src.in_features, self.q, bias=src.bias is not None)
        with torch.no_grad():
            self.lin.weight.copy_(src.weight.repeat(self.q, 1))
            if src.bias is not None:
                self.lin.bias.copy_(src.bias.repeat(self.q))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, H, F) -> (B, H, Q), quantiles tries.

        Rien ne garantit la monotonie : un croisement (q=0.3 au-dessus de
        q=0.7) est possible et connu dans la litterature. On trie a la sortie,
        ce qui ne peut qu'ameliorer la perte pinball et evite un CRPS defini
        sur une "distribution" non monotone.
        """
        return torch.sort(self.lin(x), dim=-1).values


class HeadTap:
    """Capture l'entree de la couche de sortie, sans toucher au forward.

        with HeadTap(model, "out_proj") as tap:
            point = model(x, x_mark, y_mark)   # backbone gele, no_grad
        feats = tap.value                      # (B, 48, 32), detache

    Le modele tourne inchange, donc sa prevision ponctuelle reste disponible
    comme controle a chaque step, et on n'a pas besoin de connaitre
    l'interieur du backbone. L'entree captee est detachee : rien ne remonte.
    """

    def __init__(self, model: nn.Module, head_name: str = DEFAULT_HEAD):
        self.module = dict(model.named_modules())[head_name]
        self.value: Optional[torch.Tensor] = None
        self._h = None

    def __enter__(self):
        def hook(_m, inp, _out):
            self.value = inp[0].detach()
        self._h = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
        return False
