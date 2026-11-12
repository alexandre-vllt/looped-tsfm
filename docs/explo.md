# explo.md — les pistes à tester

Créé le 23 octobre 2026. Compagnon de `done.md` (ce qui est fait) et de
`train_v2.md` (la pipeline). Ici : ce qui reste à essayer, pourquoi, et la
manière la moins chère de trancher.

Chaque entrée a la même forme : **la question**, **ce qu'on sait déjà**,
**l'expérience décisive**, **le coût**, **ce qui compterait comme résultat
positif**. Cette dernière ligne est importante — elle s'écrit *avant* le run,
sinon on lit le résultat après coup et on se raconte une histoire.

## Le plancher de bruit, à garder en tête partout

| périmètre | bruit | mesuré sur |
|---|---|---|
| GIFT-Eval test, 97 lignes | **0,77 %** | BASE contre SEED1, même recette, seeds différentes |
| GIFT-Eval validation | 1,68 % | idem, mais une fenêtre par série au lieu de 20 |
| le papier, 4 seeds | 0,18 % | leur Table 8 — ce qui est atteignable, pas ce qu'on a |

Un effet sous 0,77 % sur le test n'est pas distinguable d'un changement de
seed. Le test de signe sur 97 lignes complète la lecture : un gain petit mais
réparti sur 60+ lignes est crédible, un gain porté par trois datasets ne l'est
pas.

---

# 1. β = 1/√k — la profondeur achète-t-elle du temps de flot ?

## La question

L'itération bouclée s'écrit

    h[j+1] = β·f(h[j]) + (1-β)·h[j] = h[j] + β·( f(h[j]) - h[j] )

soit **exactement Euler explicite de pas β** sur l'EDO `ḣ = f(h) - h`. Donc k
itérations de pas β parcourent un temps total **T = k·β**.

| paramétrisation | T |
|---|---|
| β = 1/k | **1, constant** |
| β = 1/√k | √k |
| β constant | k·β |

**Les quatre runs qu'on a sont tous à β = 1/k.** Ils approchent donc tous le
*même* objet : k=1 est un pas d'Euler de taille 1, k=8 est huit pas de 1/8 du
même flot jusqu'au même temps 1. La profondeur n'achète que de la précision de
discrétisation, **pas d'expressivité**.

## Ce qu'on sait déjà

La courbe en k, chaque bras entraîné à son propre k avec β = 1/k :

| | MASE (97) | vs R0 | p (signe) |
|---|---|---|---|
| k=1 — R0 | 1,1982 | — | — |
| k=2 — R2 | 1,1996 | −0,12 % | 0,69 |
| **k=4 — R1** | **1,1729** | **+2,11 %** | **0,0005** |
| k=8 — R3 | 1,1864 | +0,99 % | 0,10 |

Un optimum net à k=4, et k=8 *moins bon* alors qu'il discrétise mieux. La
lecture EDO explique la forme : le gain de k=1 à k=4 est la résorption de
l'erreur d'Euler, et au-delà il n'y a plus rien à gagner sur un flot de temps
fixe. Le recul à k=8 vient des dynamiques d'entraînement (BPTT à travers huit
copies), pas du flot.

**Donc β = 1/k est peut-être la raison pour laquelle la courbe est plate.**

## L'expérience décisive

**k=4, β=0,5** — soit exactement 1/√4. Un seul run, **deux comparaisons à une
seule variable** :

- contre **R1** (k=4, β=0,25) : β varie, k fixe → l'effet de l'amortissement
- contre **R2** (k=2, β=0,5) : k varie, β fixe → **l'effet de la profondeur
  seule**

La seconde vaut à elle seule le run. Tous nos résultats sur k portent
l'avertissement « k et β bougent ensemble, donc un écart ne s'attribue pas à
la profondeur ». Ce bras le lève.

Variante plus forte mais plus chère : **k=8, β=0,354** (= 1/√8), même k que R3,
seul β change. ~32 h contre ~16 h, et une seule comparaison.

## Coût

~16 h sur une A100-40GB (R1 à k=4 tournait à 2,78 M pts/s contre 11,19 à k=1).
Mémoire ≈ 3,68·k + 0,6 Go, donc ~15,4 Go à k=4 : large.

## Ce qui compterait comme positif

- **β** : k=4 β=0,5 bat R1 de plus de 0,77 % → l'amortissement 1/k est
  trop fort, et la famille β est un axe de réglage à part entière.
- **k** : k=4 β=0,5 bat R2 (k=2, même β) de plus de 0,77 % → **la profondeur
  aide par elle-même**, indépendamment de β. C'est le résultat qu'on cherche
  depuis le début du projet.
- Si les deux écarts sont sous le bruit : la profondeur ne fait rien à β fixe,
  et le +2,11 % de R1 est un effet de β, pas de k. Ce serait un résultat
  négatif **intéressant** et publiable tel quel.

---

# 2. k adaptatif à l'inférence — **FERMÉ le 30 octobre**

> **Conclusion : la marge est nulle. Ne pas implémenter.**
>
> L'oracle de k — le minimum par ligne sur k ∈ {1,2,3,4,6,8}, donc une borne
> supérieure qu'aucune règle ne peut battre — donne **1,13 %** contre le
> meilleur k fixe. À peine plus que le plancher de bruit de 0,77 %.
>
> Et le chiffre décisif est la distribution du k oracle : **quasiment uniforme
> sur les six valeurs** (18,6 / 19,6 / 12,4 / 15,5 / 16,5 / 17,5 %). Si le
> meilleur k dépendait d'une propriété de la série, elle serait concentrée. Un
> étalement plat veut dire que l'argmin par ligne choisit du **bruit** — et
> prendre le minimum de six mesures bruitées donne toujours un gain positif,
> même si les six k sont équivalents.
>
> Médianes par terme : 4, 4, 3. Aucune variation exploitable.
>
> Détail dans `done.md` §5.10. La porte apprise d'Ouro ne vaut donc pas d'être
> implémentée **pour la précision** ; elle pourrait encore valoir pour le
> **compute** (même score à k moyen plus bas), ce qui est un autre résultat.
>
> Ce qui suit est conservé pour la lecture d'Ouro et les corroborations, qui
> restent valables et citables.

# 2bis. k adaptatif à l'inférence — la recette d'Ouro (archivé)

## La question

Le bloc étant à poids partagés, un même checkpoint se score à n'importe quel k
sans réentraînement. Faut-il le même k pour toutes les séries ?

Ouro (*Scaling Latent Reasoning via Looped Language Models*, ByteDance Seed,
`papers/Ouro.pdf`) répond à exactement cette question, avec un mécanisme publié
et un entraînement en deux temps. Inutile d'improviser : on a une recette.

## D'abord, deux corroborations d'Ouro sur nos propres résultats

Avant même le mécanisme, Ouro confirme indépendamment, sur une modalité
totalement différente, les deux choses qu'on a mesurées.

**Leur optimum de profondeur est 4, et l'extrapolation dégrade.** Tables 10 et
11, Ouro-1.4B évalué de 1 à 8 pas récurrents, entraîné à 4 :

| pas | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| MMLU (5-shot) | 41,2 | 60,4 | 66,7 | **67,5** | 66,6 | 65,8 | 65,3 | 64,5 |

« Performance peaks at the trained depth (T = 4) and then degrades. » C'est
exactement la forme de notre balayage de k sur R1 : optimum au k
d'entraînement, dégradation au-delà.

**Et ils sont passés de 8 à 4 pour cause d'instabilité.** §4.3.1 : ils
démarrent à 8 pas récurrents, rencontrent des instabilités, réduisent à 4, et
notent que 4 s'est révélé « optimal for the stability-performance trade-off ».
Notre courbe entraînée dit la même chose — k=8 (+0,99 %) est moins bon que k=4
(+2,11 %) — et on l'avait attribué aux dynamiques du BPTT à travers huit
copies. Ouro l'observe aussi, à une tout autre échelle.

Ces deux points valent d'être cités dans le rapport : notre résultat principal
a une confirmation externe.

**Piège de notation.** Le β d'Ouro est le coefficient d'entropie / KL de leur
objectif. Notre β est l'amortissement de l'itération, qui correspond à leur η
et au η de SFPRM. Ne jamais mélanger les deux dans le rapport.

## Le mécanisme d'Ouro

Une **porte de sortie** tourne en parallèle de la tête de prédiction, à chaque
itération t :

    λ_t(x) = σ( Linear_φ( h^(t) ) )        probabilité de sortir à l'étape t
    S_t(x) = Π_{j≤t} ( 1 - λ_j(x) )        survie : ne pas être sorti avant t
    p_φ(t|x) = λ_t(x) · S_{t-1}(x)         loi sur l'étape de sortie

avec toute la masse restante affectée à T_max, pour que ce soit une vraie loi.
Sur nano, `Linear_φ` fait d_model = 128 paramètres sur 190 000 : négligeable.

**À l'inférence — le critère Q-exit** (de PALBERT) :

    t_exit(x) = min{ m : CDF(m|x) ≥ q }

`q` est un **réglage de déploiement**, pas d'entraînement : il déplace le
compromis compute/précision sans réentraîner. C'est précisément ce qu'on veut.

## Les deux temps de l'entraînement

**Étape I — pendant le pré-entraînement (§3.3).** La porte est apprise
conjointement avec le modèle, sur un objectif régularisé par l'entropie :

    L = Σ_t p_φ(t|x) · L^(t)  -  β · H( p_φ(·|x) )

Le premier terme est la perte de tâche **pondérée par la probabilité de sortie**.
Le second empêche l'effondrement : sans lui, les itérations profondes baissent
la perte, donc attirent la masse de p_φ, donc reçoivent plus de signal, donc
baissent encore — et p_φ s'écrase sur t = T_max. Ouro montre que l'objectif est
un ELBO dont le prior sur l'étape de sortie est uniforme, le terme d'entropie
étant le KL à ce prior. Prior uniforme choisi délibérément : un prior
géométrique favoriserait les sorties précoces, l'uniforme est sans biais de
profondeur.

**Ce que ça change pour nous est petit.** On a déjà `multi_exit_loss` dans
`looped.py`, avec `--exit-loss uniform` : c'est la somme des pertes par sortie
à poids **uniformes**. L'étape I d'Ouro est la même chose à poids **appris**,
plus le terme d'entropie. Donc : ajouter la porte, remplacer les poids
uniformes par p_φ, ajouter H. Ce n'est pas une nouvelle architecture.

**Étape II — porte seule, modèle gelé (§3.4).** On gèle le modèle et on
n'entraîne que φ, sur un signal glouton construit à partir du gain réellement
obtenu par une itération de plus :

    I^(t)  = max(0, L^(t-1)_stop - L^(t)_stop)      gain, perte DETACHEE
    w^(t)  = σ( k·(I^(t) - γ) )                    k = 50, γ = 0,005
    L^(t)_adaptive = BCE( 1 - λ^(t) , w^(t) )       moyennée sur t ≥ 2

`w^(t) ≈ 1` dit « continue », `≈ 0` dit « sors ». La perte pénalise les deux
échecs symétriques : sortir quand il fallait continuer, et continuer quand il
fallait sortir.

## L'expérience décisive, et elle est gratuite

Avant tout mécanisme, mesurer **s'il y a de la marge**.
`code/outputs/eval_comparison_per_dataset.csv` contient déjà les 97 lignes pour
les six k de R1. Donc, sans toucher un GPU :

- MASE à **k fixe = 4**, le meilleur choix fixe : 1,1729
- MASE à **k oracle par ligne** : pour chaque ligne, le min sur
  k ∈ {1,2,3,4,6,8}

L'écart est la **borne supérieure de tout ce qu'une règle adaptative peut
rapporter** — aucune règle, même parfaitement informée, ne battra l'oracle.

- moins de ~1 % → abandonner la piste, acquis en dix minutes de Python
- 2 à 4 % → c'est la direction la plus prometteuse du projet

Et le second regard vaut autant : **de quoi dépend le k oracle ?** S'il corrèle
avec le terme, la fréquence ou l'horizon, la règle est une table de
correspondance. S'il a l'air aléatoire, aucune règle réalisable ne le capturera.

Point d'appui : `cka[t0_t1] = 0,623` dans les diagnostics de V2A au step 400,
contre 0,9999 mesuré à l'init sur le checkpoint k=1. Les itérés se ressemblent
beaucoup moins une fois le modèle entraîné à boucler — s'ils faisaient tous la
même chose il n'y aurait rien à adapter.

**À vérifier aussi** : quel β a utilisé le balayage de k à l'inférence ? S'il a
gardé β = 0,25 fixe en faisant varier k, ces six points ne sont pas à 1/k et ne
se lisent pas comme la courbe entraînée.

    grep -h -o "trained k=[0-9]* beta=[0-9.]*  -> EVAL k=[0-9]* beta=[0-9.]*" \
      $PROJ/results/R1_k*/eval.log | sort -u

## Ensuite, les trois stratégies d'Ouro, du moins cher au plus cher

Ouro en compare exactement trois (§5.4.1), ce qui nous donne l'ordre à suivre.

**(a) Seuil sur la différence d'état caché. AUCUN entraînement.** Sortir quand
‖h_t − h_{t−1}‖₂ < ε. C'est leur heuristique de référence, et chez nous la
quantité est **déjà calculée** — `update_magnitude` dans les diagnostics de
`looped.py`. Il suffit d'un arrêt par échantillon dans le forward et d'un
balayage de ε à l'éval. **À faire en premier**, sur le checkpoint R1 qu'on a
déjà payé.

**(b) Étape II sur R1 gelé.** Le modèle est figé, donc on n'entraîne que 128
paramètres : quelques milliers de steps, pas de réentraînement. Beaucoup moins
cher que l'étape I. Réserve honnête : Ouro fait l'étape II sur une porte déjà
pré-entraînée à l'étape I. Partir d'une porte neuve sur un modèle qui n'a
jamais vu de porte est une variante de leur recette, pas leur recette — à
présenter comme telle, et leur figure §5.4.1 compare justement une porte non
entraînée, donc il y a de quoi se situer.

**(c) Étape I complète.** La porte apprise conjointement, avec l'entropie. Un
réentraînement, ~16 h à k=4. C'est la version fidèle, et la seule qui puisse
dépasser les deux autres, mais c'est la plus chère.

## Coût

(a) une demi-journée d'écriture + une éval de 2 h, sur un checkpoint existant.
(b) ~1 à 2 h de GPU plus l'écriture de la perte adaptative.
(c) ~16 h plus les modifications de `looped.py` et `train_loop.py`.
Étape 0, l'oracle : **zéro GPU, ~10 min.**

## Ce qui compterait comme positif

- l'oracle dépasse k=4 fixe de plus de 2 % **et** le k oracle corrèle avec une
  variable observable → une règle sans coût capture une part réelle de l'écart
- **(a) ou (b) atteint la MASE de k=4 à un k moyen ≤ 2,5** → compute adaptatif
  démontré. C'est l'argument central des modèles bouclés, et « même score pour
  60 % du compute » se défend très bien, même sans gain de précision
- si rien ne bat k=4 fixe : la platitude de 0,34 % entre k=2 et k=6 était le
  vrai signal, et il n'y a rien à adapter. Résultat négatif propre, et il
  s'appuie sur l'oracle, donc il est définitif et pas seulement « on a essayé »

---

# 3. Doubler le synthétique et continuer le pré-entraînement

## La question

Le synthétique est **illimité** : on le génère. Si c'est sa quantité qui borne
le modèle, c'est le levier le moins cher de tout le projet. Si ce n'est pas
elle, on l'aura su une fois pour toutes et on arrêtera d'y penser.

Formulée précisément : R0 a tiré ~38,4 M fenêtres synthétiques (150 000 steps
x 512 x p_real 0,5) dans **825 000 séries**, soit **47 passages par série**.
Est-ce que continuer sur des séries *neuves*, jamais vues, fait repartir la
performance ?

## Ce qu'on sait déjà, et pourquoi la question n'est pas réglée

**Le corpus synthétique actuel** : `$WORK/synth`, 825 000 séries x 2304,
1,90 G points, 7,08 Go, neuf générateurs TempoPFN (anomaly, forecast_pfn, gp,
kernel, ou_process a poids 3, sawtooth, sinewave, spike, step).

**L'échelle synthétique est monotone chez nous** : 50 % de synthétique bat
25 %, qui bat 5 %. Mais c'est une question de *proportion*, pas de *quantité* —
ces trois bras puisaient dans les mêmes 825 000 séries.

**Et la montée en données RÉELLES a donné peu** : D1 a doublé les points réels
(x2,13) pour +0,79 %, soit le plancher de bruit. Si le synthétique se comporte
pareil, doubler rapportera ~0,8 % et ne vaudra pas le disque. Mais rien ne dit
qu'il se comporte pareil : le réel était déjà vu ~1 fois par série, le
synthétique 47 fois. **C'est justement le régime où la répétition peut mordre.**

À noter aussi : le papier annonce 1-5 % de synthétique, nous sommes à 50 %. On
est déjà très loin de leur recette sur cet axe.

## L'expérience décisive — un fork, deux branches

La formulation naïve « continuer R0 sur du neuf et voir si ça monte » est
**confondue avec le compute** : la branche continuée reçoit plus de steps que
R0, donc elle sera meilleure même si la fraîcheur des données n'y est pour
rien.

La forme propre est un fork à deux branches de longueur égale :

1. partir d'un **snapshot de R0 en phase STABLE** — pas de `step150000.pth`,
   qui est post-decay avec un LR à zéro : y revenir demanderait de réchauffer
   le LR, une opération délicate et un facteur de plus. R0 décaye à 112 500
   (`stable 0,75`), donc `step110000.pth` est le dernier point stable.
2. **branche A** : +50 000 steps sur le corpus synthétique ACTUEL
3. **branche B** : +50 000 steps sur un corpus synthétique NEUF, même taille,
   même mélange de générateurs, **graine différente**
4. décayer les deux identiquement, puis évaluer

L'écart A/B est alors **purement l'effet de la fraîcheur des données**, à
compute, recette, corpus réel et architecture identiques.

Et la branche A n'est pas du gaspillage : c'est elle qui dit combien rapportent
50 000 steps de plus tout court, ce qu'on ne sait pas non plus.

## Une variante plus tranchante, à considérer

« Deux fois plus de tirages de la même distribution » et « deux fois plus de
diversité » sont deux expériences différentes, et la seconde est généralement
celle qui compte. Échantillonner davantage d'une distribution fixe sature vite ;
élargir la distribution, non.

Donc si le budget ne permet qu'un seul corpus neuf, générer **825 000 séries
avec des hyperparamètres élargis** (banque de noyaux plus large, plages de
fréquence et d'amplitude étendues, voire un dixième générateur) est sans doute
plus informatif que 825 000 tirages de plus à l'identique. Le défaut : ça
change deux choses à la fois (quantité et distribution), donc un écart ne
s'attribue plus. À trancher selon ce qu'on veut apprendre — et dire lequel des
deux on a fait.

## Coût

**Génération** : ~7,1 Go de plus, machinerie existante (`code/gen_all.sh`,
`code/gen_cpu.sh`). Vérifier le disque avant : 120 Go libres au 23 octobre,
contre 178 le 17 — ça descend.

**Entraînement** : 50 000 steps a k=1 ≈ **1 h 20** par branche (150 000 steps
font ~4 h a 11,19 M pts/s). Les deux branches en parallèle sur deux GPU, plus
le decay : **moins de 3 h de mur pour les deux.** C'est de loin l'expérience la
moins chère de ce document.

**Évaluation** : deux évals GIFT-Eval, ~2 h, parallélisables.

## Ce qui compterait comme positif

- **B bat A de plus de 0,77 %** → la répétition du synthétique bornait le
  modèle, et comme le synthétique est illimité, c'est immédiatement
  exploitable : on régénère et on rallonge.
- **A ≈ B** → la quantité de synthétique n'est pas la contrainte. Négatif, mais
  précieux : ça ferme définitivement un axe qu'on aurait pu creuser longtemps,
  et ça renforce la conclusion de D1 — le trou résiduel de ~8 % est dans la
  recette, pas dans le volume de données.
- **A et B battent tous deux R0 à 150 k de façon nette** → le vrai message est
  que 150 000 steps est trop court, ce qui rejoint ce que la courbe de
  validation de V2A dira de son côté. Les deux mesures se recoupent alors, et
  c'est une conclusion plus solide que chacune prise seule.

## Dépendance

Rien ne bloque : R0 et ses snapshots existent, le générateur existe. À lancer
dès qu'un GPU est libre.

---

# 4. Perte quantile — la moitié du benchmark qu'on ne joue pas

## La question

Notre modèle produit une **prévision ponctuelle** : 48 points, entraînés par
MAE (`reverso_loss`, papier §2.1). Mais GIFT-Eval note aussi des métriques
**probabilistes** — CRPS et `mean_weighted_sum_quantile_loss` sont dans la
liste `METRICS` de notre propre `eval_gift.py`.

Et voilà comment on les obtient aujourd'hui :

    samples = np.repeat(pred_full_np[i][None, :], self.num_samples, axis=0)

**Cent copies identiques de la prévision ponctuelle.** Tous les quantiles sont
égaux, la distribution prédite est un Dirac, et le CRPS dégénère. On soumet un
point déguisé en distribution. La moitié probabiliste du benchmark n'est pas
jouée — elle est remplie avec du remplissage.

## Ce que fait le papier, et c'est bon marché

Reverso §C.5, *Probabilistic Forecasting* :

> « we repurpose a pretrained Reverso model by **training the decoder head with
> a quantile regression objective** […] this experiment does not train the full
> model from scratch for probabilistic forecasting »

Donc : **backbone gelé, on ne réentraîne que la tête.** C'est une recette
publiée, et la moins chère de tout ce document — pas de pré-entraînement, pas
de BPTT à travers les boucles, juste une tête `Linear(2048 -> 48)` remplacée
par `Linear(2048 -> 48 x Q)` et une perte pinball.

Leur Table 7, CRPS, Reverso 3M :

| Split | TiRex 30M | FlowState 9.1M | Moirai-2 11M | TempoPFN 30M | **Reverso 3M** |
|---|---|---|---|---|---|
| Full | 0,489 | 0,502 | 0,516 | 0,533 | **0,499** |
| Short | 0,502 | 0,519 | 0,516 | 0,547 | 0,526 |
| Medium | 0,474 | 0,485 | 0,519 | 0,543 | **0,459** |
| Long | 0,467 | 0,477 | 0,513 | 0,539 | 0,470 |

Avec une tête quantile, Reverso 3M devient **le meilleur du tableau sur le
split medium**. C'est le seul endroit du papier où il bat tout le monde.

## Ce qu'on sait déjà, et un fait jamais remarqué

Notre colonne CRPS dans `aggregate_eval.py` :

| bras | MASE | CRPS |
|---|---|---|
| R0_k1 | 1,1982 | 0,2825 |
| **R1_k4** | **1,1729** | **0,2846** |
| R3 | 1,1864 | 0,2823 |
| D1 | 1,1887 | 0,2859 |
| NANO_pub | 1,0742 | 0,2712 |

**R1 est notre meilleur bras en MASE et notre PIRE en CRPS**, pire que R0.
Personne n'avait regardé cette colonne. Deux lectures possibles :

- avec une distribution dégénérée, CRPS n'est qu'une MAE redéguisée, et
  l'inversion est du bruit — à vérifier, c'est gratuit
- ou le bouclage améliore le point central en abîmant quelque chose que le CRPS
  capte et la MASE non

Dans les deux cas, **on ne peut pas trancher avec un Dirac**. Il faut une vraie
distribution pour que la colonne CRPS veuille dire quelque chose.

## L'expérience décisive

1. Remplacer la tête par `Linear(output_bottleneck_dim -> 48 x Q)` avec
   Q = 9 quantiles, les mêmes que `MeanWeightedSumQuantileLoss` du harnais :
   0,1 à 0,9.
2. **Geler le backbone**, n'entraîner que la tête, perte pinball.
3. À l'inférence, remplacer le `np.repeat` par les quantiles prédits.
4. Évaluer CRPS et MASE — la MASE doit se lire sur la médiane (q = 0,5), donc
   elle est aussi un contrôle : si elle s'effondre, la tête est mal entraînée.

À faire sur **R1**, notre meilleur bras, et sur **R0** comme référence. Deux
têtes, deux comparaisons.

Un piège à éviter : le rollout autorégressif réinjecte la prévision. Avec une
tête quantile il faut décider **quel quantile est réinjecté** — la médiane est
le choix naturel, et c'est ce qui garde la MASE comparable. À écrire
explicitement dans le code, sinon quelqu'un réinjectera q=0,5 par accident en
croyant réinjecter la moyenne.

## Coût

La tête fait `48 x 128` paramètres, le backbone est gelé : **quelques milliers
de steps, moins d'une heure**. Plus une éval GIFT-Eval de 2 h par tête.
C'est, avec l'oracle de k, l'expérience la moins chère du document.

## Ce qui compterait comme positif

- **le CRPS s'améliore nettement** — il part d'un Dirac, donc presque n'importe
  quelle distribution calibrée devrait gagner. Un gain ici est quasi acquis,
  ce qui en fait moins une découverte qu'un **trou à boucher** : aujourd'hui on
  rapporte un CRPS qui ne veut rien dire.
- **l'inversion R0/R1 en CRPS se résout** dans un sens ou dans l'autre. Si R1
  reste pire en CRPS avec une vraie distribution, c'est un résultat : le
  bouclage améliore le point central et dégrade la calibration, ce qui n'est
  documenté nulle part.
- **la MASE sur la médiane ne se dégrade pas** — sinon la tête quantile coûte
  en précision ponctuelle ce qu'elle gagne en calibration, et il faut le dire.

## Pourquoi c'est haut dans la liste malgré un résultat « acquis »

Parce que le rapport contient actuellement une colonne CRPS pour dix-huit bras,
et que cette colonne mesure un Dirac. Soit on la répare, soit on l'enlève du
rapport en expliquant pourquoi. La réparer coûte une heure.

---

# 5. Complex KDA — donner au récurrent le droit de tourner

<https://arxiv.org/abs/2609.24797>

## MISE À JOUR du 2 novembre — la moitié de ce point est déjà dans notre code

En écrivant le patch, j'ai trouvé ceci dans `reverso_torch/model.py` :

    265     if self.allow_neg_eigval:
                beta = beta * 2.0

`allow_neg_eigval` vient de la littérature DeltaNet et fait **exactement** la
moitié de ce qui est proposé ci-dessous : β dans (0,2) au lieu de (0,1). Il est
là depuis le début, codé en dur à `False` en trois endroits, exposé nulle part
— ni dans `train_loop.py`, ni dans `args.json`, ni dans `eval_gift.py`. Et
`fla.DeltaNet` accepte le même drapeau, donc **le risque que le noyau clampe
silencieusement n'existe pas** : c'est une voie supportée des deux côtés.
L'étape 0 que j'annonçais comme un test de faisabilité n'en est pas un, c'est
directement l'expérience.

Mais l'autre moitié n'est pas disponible : `use_gate=False` partout, il n'y a
**aucun a_t** dans le modèle. Or c'est la porte négative qui fournit la seconde
réflexion, et deux réflexions font une rotation. Donc ce point se coupe en deux,
et il faut être honnête sur ce que chaque moitié donne :

- **5a — β dans (0,2).** Un drapeau, ~4 h, lançable aujourd'hui. La valeur
  propre 1−β de (I − βkkᵀ) devient négative, donc la transition **réfléchit**
  l'état : il gagne le droit de changer de signe. C'est de l'oscillation à la
  fréquence de Nyquist, **pas** une rotation d'angle quelconque. C'est moins que
  l'article, et ça doit être présenté ainsi.
- **5b — la porte a_t dans [−1,1].** Projection neuve plus noyau gaté. Hors
  périmètre du semestre. À citer comme suite, pas à tenter.

**5a se compare à D1 (1,1887)** : même corpus v2, même recette R0, seule la plage
de β change. Et son hash de flux doit valoir **f2f87e24**, celui de D1 : le
drapeau touche au modèle, pas aux données. S'il diffère, autre chose a bougé.

Le danger réel n'est pas le noyau, c'est l'évaluation : un modèle entraîné à
β ∈ (0,2) et évalué à β ∈ (0,1) est un autre modèle, et c'est le mécanisme
exact du 17 octobre. `patch_neg_eigval.py` le rend structurellement impossible :
`swap_to_fla` lit le drapeau **sur le module** au lieu de le recevoir en
argument, et `eval_gift.py` refuse de tourner si le `--size` enregistré dans
`meta.json` ne correspond pas au répertoire de `--json_path`.

Ce qui suit reste la motivation théorique, écrite avant cette découverte.

## La question

Le cœur récurrent de Reverso est une DeltaNet : à chaque pas, l'état est
multiplié par une matrice de transition, puis une information nouvelle est
écrite. Dans la variante à portes (KDA, celle de Kimi), cette transition
s'écrit

    S_t = Diag(a_t) · (I − β_t k_t k_tᵀ) · S_{t−1} + β_t k_t v_tᵀ

avec `a_t ∈ (0,1)^d` la porte d'oubli par canal et `β_t ∈ (0,1)` le pas
d'écriture. Les deux sont des sigmoïdes, donc **positifs**. Conséquence :
toutes les valeurs propres de la transition sont réelles positives, et l'état
ne sait faire qu'une seule chose — **décroître**. Il ne peut pas osciller.

La proposition du papier est d'une simplicité désarmante : élargir les deux
plages. `a_t ∈ [−1,1]` au lieu de `(0,1)`, `β_t ∈ [0,2]` au lieu de `(0,1)`.

Pourquoi ça change la nature de l'opérateur :

- à `β = 2`, `I − 2kkᵀ` est **exactement une réflexion de Householder** :
  orthogonale, de norme 1, elle ne perd rien ;
- une entrée **négative** de `Diag(a_t)` est une seconde réflexion, sur un
  autre axe ;
- et le produit de deux réflexions est une **rotation**.

Donc l'état gagne le droit de tourner au lieu de seulement s'éteindre. Et la
stabilité est préservée gratuitement : `|a| ≤ 1` et `β ∈ [0,2]` donnent
`‖I − βkkᵀ‖₂ ≤ 1`, la transition reste **non-expansive**. On n'achète pas
l'oscillation au prix d'une divergence possible. La transition reste aussi
diagonale-plus-rang-un, donc le noyau chunké reste applicable — c'est ce qui
rend la chose bon marché.

## Pourquoi ça nous concerne particulièrement

**Parce qu'une série temporelle, c'est de l'oscillation.** La saisonnalité est
littéralement une phase qui avance. Un état purement décroissant doit encoder
une période *par la mémoire brute* — retenir la valeur d'il y a 24 pas. Un état
qui tourne l'encode **dans sa phase**, ce qui est exponentiellement moins cher.

Et nous avons un aveu dans notre propre code : Reverso traite les longues
périodicités par la convolution FFT, et quand la saisonnalité dépasse `L = 2048`
il faut un **sous-échantillonnage FFT à l'inférence** pour s'en sortir. Ce
rustine-là existe précisément parce que la partie récurrente ne porte pas bien
la périodicité.

**Et surtout, c'est le seul changement d'architecture de ce document qui soit
plausiblement _synergique_ avec le bouclage, et non orthogonal.** Boucler, c'est
appliquer `f` k fois. Composer k applications d'une map qui décroît donne une
map qui décroît — on n'apprend rien de nouveau au passage k. Composer k
rotations fait avancer la phase de k fois l'angle : chaque passage *ajoute* de
la structure. Si l'hypothèse tient, l'optimum de la courbe en k devrait se
**déplacer vers les grands k** avec des portes complexes. C'est une prédiction
falsifiable, et aucun autre point de ce document n'en fournit.

## Ce qu'on sait déjà

Rien de mesuré. C'est le point le plus spéculatif du document, et c'est assumé.
Deux faits de contexte, quand même :

- §5.9 a montré que le gain du bouclage **décroît avec l'horizon**
  (+2,41 % / +1,91 % / +1,50 % à k=4, et à k=8 il devient négatif sur medium et
  long). Le bouclage améliore un patch et se dilue dans la chaîne autorégressive.
  Si le mécanisme manquant est la périodicité, c'est exactement sur les horizons
  longs qu'un état rotatif devrait payer — là où nous perdons aujourd'hui.
- le noyau `ChunkDeltaRuleFunction` de `fla` nous a déjà refusé le float32 le
  24 octobre. Il est écrit pour des plages précises. **Rien ne garantit qu'il
  accepte `β > 1`** : il peut clamper, ou produire des NaN silencieusement.
  C'est le risque n° 1 et il est vérifiable en quinze minutes.

## L'expérience décisive

**Étape 0 — le smoke test, quinze minutes, zéro GPU-heure utile.** Avant toute
chose : élargir les deux activations, faire *un* forward, et vérifier trois
choses. (a) pas de NaN. (b) la perte au pas 0 est inchangée si on initialise
les biais pour que les nouvelles activations **reproduisent les anciennes
valeurs** — `2σ(x′) − 1 = σ(x)` se résout en `x′ = σ⁻¹((σ(x)+1)/2)`, donc le
démarrage à chaud depuis un checkpoint Reverso est **exact**, pas approximatif.
C'est important : ça veut dire que ce point peut se tester en *continuation* et
pas seulement en pré-entraînement complet. (c) le noyau ne clampe pas — le
vérifier en passant `β = 1,8` à la main et en comparant à un calcul de référence
en boucle Python sur un petit tenseur.

Si (c) échoue, le point est mort tel quel : il faudrait réécrire le noyau, ce
qui sort du périmètre d'un projet de semestre. **Le dire et passer.**

**Étape 1 — le run, ~4 h.** Recette R0 à l'identique, corpus v2, k=1, portes
élargies. Une seule variable change par rapport à D1. Lire la MASE de
validation.

**Étape 2, conditionnelle — la synergie, ~16 h.** Seulement si l'étape 1 gagne :
le même modèle à k=4, β=0,25, et on regarde si l'optimum en k s'est déplacé.
C'est *ça* le résultat intéressant, pas le gain à k=1.

## Coût

Quinze minutes pour l'étape 0, ~4 h pour l'étape 1, ~16 h pour l'étape 2.
L'étape 0 est à faire **tout de suite** parce qu'elle est gratuite et qu'elle
peut fermer le point.

Modification : deux lignes d'activation dans le module de portes, plus
l'initialisation des biais pour le démarrage à chaud. Pas de changement de
forme de tenseur, pas de changement de noyau.

## Ce qui compterait comme positif

- **étape 0** : pas de NaN, perte au pas 0 identique à 1e-4 près après
  réinitialisation des biais, et le noyau honore `β = 1,8` au lieu de le
  clamper à 1. Trois cases à cocher, écrites avant de regarder.
- **étape 1** : plus de 0,77 % de gain en MASE de validation — le plancher de
  bruit. En dessous, ce n'est pas un résultat.
- **étape 1, le signe fin qui compte vraiment** : le gain doit être **plus
  grand sur les séries saisonnières et sur le terme long** que sur le court. Un
  gain uniforme sur les trois termes voudrait dire que ça marche pour une autre
  raison que celle qu'on raconte, et il faudrait le dire.
- **étape 2** : l'optimum de la courbe en k se déplace au-delà de k=4. C'est la
  seule observation de ce document qui lierait l'architecture au bouclage au
  lieu de les additionner.

## La réserve honnête

Ce n'est pas notre contribution, c'est celle d'un papier de 2026 qu'on
appliquerait. À présenter comme tel : « nous avons testé si une transition
capable de tourner change le comportement du bouclage », et pas comme une
trouvaille. Le résultat a de la valeur dans les deux sens — un échec propre
sur une hypothèse clairement énoncée vaut mieux qu'un gain inexpliqué.

---

# 6. L'augmentation — le plus gros levier jamais testé

Aucun des dix runs n'a utilisé d'augmentation. L'Algorithme 4 du papier
(sous-échantillonnage, modulation d'amplitude, découpe, retournements,
censure, mixup) est **câblé** dans `reverso_torch/augment.py` depuis le
12 octobre et n'a jamais servi. Le papier, lui, s'en sert.

C'est très probablement une part des ~8 % de trou qu'on n'explique ni par le
compute ni par les données.

**Expérience** : recette R0 + `--augment`, corpus v2, k=1. ~4 h. Le coût
mesuré de l'augmentation était de 24,1 ms par batch, soit quelques pour cent.

**Réserve honnête** : les constantes de `CFG` dans `augment.py` sont les
miennes, pas celles du papier, qui ne les donne pas. Il faudrait d'abord
extraire les valeurs par défaut de `src/data/augmentations.py` de TempoPFN —
c'est le générateur qu'Arber nous a donné, donc c'est la meilleure
approximation disponible de leurs réglages.

**Positif** : plus de 0,77 % de gain. Et si c'est beaucoup plus, ça réoriente
tout ce qui reste.

---

# 7. Une seconde seed pour R1

Tout notre plancher de bruit repose sur **deux** runs (BASE et SEED1), ce qui
ne permet pas d'estimer un écart-type. Le papier en a quatre et annonce 0,18 %,
soit 4× moins que notre 0,77 %.

Une seconde seed de R1 donnerait la barre d'erreur du résultat principal du
projet — le +2,11 %. C'est, scientifiquement, le run le plus rentable qui
reste, et il est reporté depuis le 13 octobre.

**Coût** : ~16 h à k=4. **Positif** : l'écart entre les deux seeds de R1 est
nettement sous 2,11 %, ce qui consolide le résultat principal au lieu de le
laisser reposer sur un seul tirage.

---

# 8. Injection d'entrée (SFPRM)

Notre itération n'injecte pas l'entrée :

    nôtre   h[j+1] = β·f(h[j])    + (1-β)·h[j]
    SFPRM   h[j+1] = η·f(h[j], x) + (1-η)·h[j]

L'injection donne l'**existence** d'un point fixe (Brouwer sur la boule), ce
que notre forme ne garantit pas. Le Théorème 1 de SFPRM donne en plus une
fenêtre de stabilité explicite sur η, qui est identiquement notre β.

C'était la décision n° 4 du projet, prise *contre* l'injection. La revisiter
est un changement d'architecture, pas un réglage — donc à ne lancer que si
1, 3 ou 4 ne donnent rien, et en le présentant comme une variante et non comme
« notre modèle ».

**Coût** : ~16 h plus une modification de `looped.py`. Lie directement le
projet au papier d'Arber, ce qui a de la valeur en soi.

---

# 9. Gratuit, à faire de toute façon

Ces trois-là ne demandent aucun GPU et sortent des CSV déjà écrits.

**La décomposition par terme de la courbe en k.** Le bouclage aide-t-il plus
sur les horizons longs ? Le benchmark mesure une *chaîne* de patches
autorégressifs, le held-out un seul patch — c'est très probablement là que se
cache pourquoi nos gains held-out ne se transfèrent pas. Alimente directement
la piste 2.

**L'oracle de k** (voir piste 2, étape 1).

**La liste exacte du sous-ensemble du papier.** Nos 63 lignes sont une
reconstitution à partir des 21 préfixes portant les trois termes, et elle
donne un décalage systématique d'environ 0,011 contre les chiffres d'octobre.
Une question aux auteurs solidifierait tous les nombres du sous-ensemble.

---

# Ordre proposé, sous contrainte de temps

| | piste | coût | pourquoi là |
|---|---|---|---|
| 1 | **oracle de k + décomposition par terme** (piste 9) | 0 GPU, ~30 min | gratuit, et décide si la piste 2bis vaut quoi que ce soit — **fait, les deux ferment la piste 2** |
| 2 | **smoke test Complex KDA** (piste 5, étape 0) | ~15 min | gratuit, et peut fermer le point sur un refus du noyau |
| 3 | **tête quantile** (piste 4) | ~1 h, 297 paramètres | répare une colonne CRPS qui mesure aujourd'hui un Dirac sur dix-huit bras |
| 4 | **fork synthétique frais contre réutilisé** (piste 3) | ~3 h | ferme ou ouvre un axe entier pour trois heures |
| 5 | **augmentation** (piste 6) | ~4 h | le plus gros levier jamais testé, et déjà câblé |
| 6 | **Complex KDA, run complet** (piste 5, étape 1) | ~4 h | seulement si l'étape 0 passe |
| 7 | **k=4, β=0,5** (piste 1) | ~16 h | deux comparaisons à une variable, dont « la profondeur seule » |
| 8 | **seconde seed de R1** (piste 7) | ~16 h | met une barre d'erreur sur le résultat principal du projet |
| 9 | Complex KDA à k=4 (piste 5, étape 2) | ~16 h | la seule mesure de synergie entre architecture et bouclage |
| 10 | seuil sur ‖h_t − h_{t−1}‖ (piste 2bis, (a)) | ~½ journée | **gelé** : l'oracle ne laisse que 1,13 % de marge, soit le bruit |
| 11 | injection d'entrée (piste 8) | ~16 h | seulement si le reste ne donne rien |

Les lignes 4 à 8 tiennent **en parallèle** : sept GPU sont libres pendant qu'un
seul run occupe le zéro. Le mur total est celui du plus long, pas la somme.

Les étapes gratuites (1, 2) et les pistes qui peuvent fermer un axe entier (3,
4) passent devant celles qui ne peuvent que l'améliorer — un axe fermé est du
temps rendu, et c'est le seul rendement garanti du document.

**Ce qui n'est pas dans cette table, et qui devrait passer avant tout :
écrire.** Six résultats sont acquis et non rédigés — l'optimum de la courbe en
k, « la profondeur d'entraînement bat la profondeur d'inférence », le gain qui
décroît avec l'horizon, la décomposition du trou de 10,35 %, la fermeture du k
adaptatif par l'oracle, et la pipeline v2 qui bat la v1 à k=1. Aucun run de
cette table ne vaut autant qu'un de ces six paragraphes.
