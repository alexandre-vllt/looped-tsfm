#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Branche la tete quantile dans l'evaluation GIFT-Eval.

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ
    $VENV/bin/python patch_qhead_eval.py --check
    $VENV/bin/python patch_qhead_eval.py
    $VENV/bin/python patch_qhead_eval.py --revert

CE QUE CA REPARE
example/eval_gift.py fabrique aujourd'hui ses echantillons ainsi :

    samples = np.repeat(pred_full_np[i][None, :], self.num_samples, axis=0)

Cent copies de la prevision ponctuelle. Le CRPS que nous rapportons depuis le
debut, pour DIX-HUIT bras, mesure donc un Dirac -- c'est-a-dire une MAE
normalisee, et rien d'autre.

LES TROIS PIEGES, ET COMMENT ILS SONT TRAITES

1. L'ESPACE. model.py 384-407 normalise x en min-max PUIS, apres out_proj,
   refait dec_out * x_range + x_min. La tete quantile reproduit out_proj, donc
   elle predit du NORMALISE. C'est exactement le bug qui a fait lire 1.27e7 au
   pas 0 a l'entrainement le 3 novembre. Ici on recalcule x_min et x_range
   depuis x_in A CHAQUE PAS DU ROLLOUT -- la fenetre glisse, donc les
   statistiques changent -- avec le meme clamp 1e-5 que model.py.

2. LA RETROACTION. Le rollout remet sa propre prediction en entree. Avec une
   tete quantile il faut reinjecter la MEDIANE, pas la moyenne des quantiles ni
   un niveau quelconque. --qhead-feedback point permet de reinjecter la tete
   ponctuelle d'origine : la chaine est alors identique a celle des dix-sept
   autres bras, ce qui isole l'effet sur le CRPS seul. Les deux coincident a
   l'initialisation, donc l'ecart entre les deux reglages mesure exactement
   l'asymetrie apprise par la tete. Lancer les deux coute deux heures et repond
   a une question qu'on se poserait de toute facon.

3. LA FORME RENDUE. On rend un QuantileForecast et non cent echantillons.
   DEFAULT_QUANTILES vaut (0.1 ... 0.9), soit exactement les neuf niveaux que
   demande MeanWeightedSumQuantileLoss dans METRICS : les cles correspondent,
   aucune interpolation. On ajoute la cle "mean" avec la prevision ponctuelle
   pour que MSE(forecast_type="mean") reste comparable aux autres bras.

   RESERVE : MSIS demande les niveaux 0.025 et 0.975, que nous n'avons pas.
   gluonts prendra les plus proches (0.1 et 0.9), donc la colonne MSIS de ce
   bras n'est PAS comparable aux autres. A dire dans le rapport ou a retirer.

4. LE RETOURNEMENT. --force-flip-invariance moyenne la prevision avec sa
   version sur -x. Sur une distribution, retourner echange q et 1-q, donc la
   moyenne n'a pas de sens. Le script REFUSE la combinaison.

ET UN GARDE-FOU DE PLUS
La tete porte le k et le beta auxquels elle a ete entrainee. Si on l'evalue a
d'autres, le script s'arrete : une tete accrochee a R1 a k=4 n'est pas une tete
de R0. C'est le meme accident que le 17 octobre, applique a la tete.

COMMENT LANCER
    PYTHONPATH="$WORK/gift-eval-code/src:$PYTHONPATH" $VENV/bin/python \
      example/eval_gift.py \
      --checkpoint runs/R1/step150000.pth \
      --json_path checkpoints/reverso_nano/args.json \
      --loop-k 4 --loop-beta 0.25 \
      --qhead runs/R1/qhead.pth \
      --output-dir results/R1Q --dataset m4_weekly

Faire CE dataset unique d'abord, deux minutes. Ce qu'il faut y lire : la ligne
"tete quantile : 9 niveaux sur out_proj", et un CRPS qui DIFFERE de celui de
R1 (0.2846). S'il est identique, la tete n'est pas branchee et le Dirac est
toujours la.
"""
import ast
import io
import os
import shutil
import sys

PROJ = os.environ.get("PROJ") or os.path.dirname(os.path.abspath(__file__))
GIFT = os.path.join(PROJ, "example", "eval_gift.py")
MARK = "self.qhead_feedback"

SUBS = [
    # -- 1. signature du predictor ------------------------------------------
    (r'''        use_ema: bool = True,
        use_fla: bool = True,
    ):''',
     r'''        use_ema: bool = True,
        use_fla: bool = True,
        qhead_path: str = "",
        qhead_feedback: str = "median",
    ):'''),

    # -- 2. attributs -------------------------------------------------------
    (r'''        self.use_ema = bool(use_ema)
        self.use_fla = bool(use_fla)''',
     r'''        self.use_ema = bool(use_ema)
        self.use_fla = bool(use_fla)
        self.qhead_path = str(qhead_path or "")
        self.qhead_feedback = str(qhead_feedback)
        self.qhead = None
        self.quantiles = ()
        self._qname = ""'''),

    # -- 3. chargement de la tete, APRES le swap fla ------------------------
    (r'''            except Exception as e:
                print(f"  fla swap skipped: {type(e).__name__}: {str(e)[:60]}")''',
     r'''            except Exception as e:
                print(f"  fla swap skipped: {type(e).__name__}: {str(e)[:60]}")

        if self.qhead_path:
            if self.force_flip_invariance:
                raise SystemExit(
                    "!! --qhead et --force-flip-invariance sont incompatibles : "
                    "retourner le signe echange q et 1-q, donc une distribution "
                    "ne peut pas etre moyennee avec sa version retournee.")
            from reverso_torch.quantile import (DEFAULT_HEAD, QuantileHead,
                                                find_head)
            blob_q = torch.load(self.qhead_path, map_location="cpu",
                                weights_only=False)
            self.quantiles = tuple(float(q) for q in blob_q["quantiles"])
            self._qname, src_lin = find_head(
                self.model, blob_q.get("head_name", DEFAULT_HEAD))
            self.qhead = QuantileHead(src_lin, len(self.quantiles))
            self.qhead.load_state_dict(blob_q["qhead"])
            self.qhead.to(self.device).eval()
            for p_ in self.qhead.parameters():
                p_.requires_grad_(False)
            k_tr = blob_q.get("loop_k")
            b_tr = blob_q.get("loop_beta")
            if k_tr is not None and int(k_tr) != int(self.loop_k):
                raise SystemExit(
                    f"!! la tete a ete entrainee a k={k_tr} beta={b_tr} et on "
                    f"evalue a k={self.loop_k} beta={self.loop_beta}. Une tete "
                    f"est accrochee a UN backbone : elle ne transfere pas.")
            print(f"  tete quantile : {len(self.quantiles)} niveaux sur "
                  f"{self._qname}, entrainee a k={k_tr} beta={b_tr}")
            print(f"  retroaction du rollout : {self.qhead_feedback}")'''),

    # -- 4. le rollout rend aussi les quantiles ------------------------------
    (r'''        for _ in range(steps):
            x_in = batch_ctx[:, -self.seq_len:, :]
            x_mark = torch.zeros_like(x_in)

            if autocast_fp is not None and self.use_amp and use_bf16:
                try:
                    with autocast_fp('cuda', dtype=torch.bfloat16):
                        outputs = self.model(x_in, x_mark, y_mark)
                except Exception:
                    outputs = self.model(x_in, x_mark, y_mark)
            else:
                outputs = self.model(x_in, x_mark, y_mark)

            out_chunk = outputs[:, -self.output_token_len:, :]
            take_chunk = out_chunk[:, :roll_len, :]
            preds.append(take_chunk)
            batch_ctx = torch.cat([batch_ctx, take_chunk], dim=1)

        return torch.cat(preds, dim=1)''',
     r'''        qpreds: List[torch.Tensor] = []

        for _ in range(steps):
            x_in = batch_ctx[:, -self.seq_len:, :]
            x_mark = torch.zeros_like(x_in)

            if self.qhead is not None:
                from reverso_torch.quantile import HeadTap
                tap = HeadTap(self.model, self._qname)
            else:
                from contextlib import nullcontext
                tap = nullcontext()

            with tap:
                if autocast_fp is not None and self.use_amp and use_bf16:
                    try:
                        with autocast_fp('cuda', dtype=torch.bfloat16):
                            outputs = self.model(x_in, x_mark, y_mark)
                    except Exception:
                        outputs = self.model(x_in, x_mark, y_mark)
                else:
                    outputs = self.model(x_in, x_mark, y_mark)

            out_chunk = outputs[:, -self.output_token_len:, :]
            take_chunk = out_chunk[:, :roll_len, :]

            if self.qhead is not None:
                # model.py 384-407 : x est normalise en min-max, et APRES
                # out_proj le modele refait dec_out * x_range + x_min. feats
                # est l'entree de out_proj, donc qhead(feats) vit dans
                # l'espace NORMALISE et doit etre ramene avec exactement les
                # memes statistiques -- recalculees ici a CHAQUE pas, parce
                # que la fenetre glisse pendant le rollout. Clamp 1e-5 comme
                # model.py 388, pas 1e-8.
                feats = tap.value
                if feats is None:
                    raise SystemExit(
                        f"!! le hook sur {self._qname} n'a rien capture.")
                xs_ = x_in.squeeze(-1)
                x_min_b = xs_.amin(1).view(-1, 1, 1)
                x_rng_b = (xs_.amax(1).view(-1, 1, 1) - x_min_b).clamp_min(1e-5)
                q_norm = self.qhead(
                    feats[:, -self.output_token_len:, :].to(torch.float32))
                q_chunk = (q_norm * x_rng_b + x_min_b)[:, :roll_len, :]
                qpreds.append(q_chunk)
                if self.qhead_feedback == "median":
                    mid = q_chunk.shape[-1] // 2
                    take_chunk = q_chunk[..., mid:mid + 1].to(take_chunk.dtype)

            preds.append(take_chunk)
            batch_ctx = torch.cat([batch_ctx, take_chunk], dim=1)

        return (torch.cat(preds, dim=1),
                torch.cat(qpreds, dim=1) if qpreds else None)'''),

    # -- 5. import gluonts ---------------------------------------------------
    (r'''        from gluonts.model.forecast import SampleForecast''',
     r'''        from gluonts.model.forecast import QuantileForecast, SampleForecast'''),

    # -- 6. appels au rollout ------------------------------------------------
    (r'''            pred_pos = self._decode_autoregressive(batch_ctx, use_bf16, downsample_factors)
            if self.force_flip_invariance:
                pred_neg = self._decode_autoregressive(-batch_ctx, use_bf16, downsample_factors)
                pred_full = 0.5 * (pred_pos - pred_neg)
            else:
                pred_full = pred_pos''',
     r'''            pred_pos, q_full = self._decode_autoregressive(batch_ctx, use_bf16, downsample_factors)
            if self.force_flip_invariance:
                pred_neg, _ = self._decode_autoregressive(-batch_ctx, use_bf16, downsample_factors)
                pred_full = 0.5 * (pred_pos - pred_neg)
            else:
                pred_full = pred_pos'''),

    # -- 7. construction des previsions --------------------------------------
    (r'''            pred_full_np = pred_full.float().squeeze(-1).detach().cpu().numpy()
            pred_list = []
            for i in range(len(downsample_factors)):
                df = downsample_factors[i]
                target_pred_len = int(self.prediction_length) // int(max(1, df))
                seq = pred_full_np[i, :target_pred_len]
                if df > 1:
                    old_len = len(seq)
                    new_len = int(self.prediction_length)
                    seq = np.interp(np.linspace(0, 1, new_len), np.linspace(0, 1, old_len), seq)
                pred_list.append(seq)
            pred_full_np = np.array(pred_list)

            for i, ts in enumerate(batch):
                start_date = ts["start"] + len(ts["target"])
                samples = np.repeat(pred_full_np[i][None, :], self.num_samples, axis=0)
                forecasts.append(SampleForecast(samples=samples, start_date=start_date))''',
     r'''            pred_full_np = pred_full.float().squeeze(-1).detach().cpu().numpy()
            q_np = (q_full.float().detach().cpu().numpy()
                    if q_full is not None else None)          # (B, T, Q)
            pred_list, q_list = [], []
            for i in range(len(downsample_factors)):
                df = downsample_factors[i]
                target_pred_len = int(self.prediction_length) // int(max(1, df))
                seq = pred_full_np[i, :target_pred_len]
                grid_new = np.linspace(0, 1, int(self.prediction_length))
                grid_old = np.linspace(0, 1, len(seq)) if len(seq) else grid_new
                if df > 1:
                    seq = np.interp(grid_new, grid_old, seq)
                pred_list.append(seq)
                if q_np is not None:
                    qi = q_np[i, :target_pred_len, :]
                    if df > 1:
                        qi = np.stack([np.interp(grid_new, grid_old, qi[:, j])
                                       for j in range(qi.shape[1])], axis=1)
                    q_list.append(qi)
            pred_full_np = np.array(pred_list)

            for i, ts in enumerate(batch):
                start_date = ts["start"] + len(ts["target"])
                if q_np is None:
                    samples = np.repeat(pred_full_np[i][None, :],
                                        self.num_samples, axis=0)
                    forecasts.append(SampleForecast(samples=samples,
                                                    start_date=start_date))
                    continue
                qi = q_list[i]                                 # (T, Q)
                bad = ~np.isfinite(qi)
                if bad.any():
                    qi = np.where(bad, pred_full_np[i][:, None], qi)
                # Les cles sont les niveaux eux-memes, donc
                # MeanWeightedSumQuantileLoss les lit sans interpoler. "mean"
                # porte la prevision ponctuelle pour que MSE(forecast_type=
                # "mean") reste comparable aux autres bras.
                arrays = np.concatenate(
                    [qi.T, pred_full_np[i][None, :]], axis=0)  # (Q+1, T)
                keys = [str(q) for q in self.quantiles] + ["mean"]
                forecasts.append(QuantileForecast(
                    forecast_arrays=arrays, start_date=start_date,
                    forecast_keys=keys))'''),

    # -- 8. les drapeaux -----------------------------------------------------
    (r'''    parser.add_argument("--no-fla", dest="use_fla", action="store_false")''',
     r'''    parser.add_argument("--no-fla", dest="use_fla", action="store_false")
    parser.add_argument("--qhead", dest="qhead", default="",
                        help="chemin d'une tete quantile entrainee "
                             "(runs/<TAG>/qhead.pth). Sans ce drapeau, le CRPS "
                             "rapporte mesure un Dirac : cent copies de la "
                             "prevision ponctuelle.")
    parser.add_argument("--qhead-feedback", dest="qhead_feedback",
                        default="median", choices=["median", "point"],
                        help="ce que le rollout autoregressif remet en entree. "
                             "median = la tete quantile est le modele. point = "
                             "la chaine reste celle des autres bras, ce qui "
                             "isole l'effet sur le CRPS seul.")'''),

    # -- 9. passage au predictor ---------------------------------------------
    (r'''                use_ema=args.use_ema,
                use_fla=args.use_fla,
            )''',
     r'''                use_ema=args.use_ema,
                use_fla=args.use_fla,
                qhead_path=args.qhead,
                qhead_feedback=args.qhead_feedback,
            )'''),

    # -- 10. le dire dans l'entete -------------------------------------------
    (r'''    print(f"allow_neg_eigval: {ALLOW_NEG}")''',
     r'''    print(f"allow_neg_eigval: {ALLOW_NEG}")
    print(f"Quantile head: {args.qhead or 'aucune -- le CRPS mesure un Dirac'}")'''),
]


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    if not os.path.exists(GIFT):
        sys.exit("!! introuvable : %s" % GIFT)
    s = io.open(GIFT, encoding="utf-8").read()

    if mode == "check":
        print("  %s  %s" % ("PATCHE    " if MARK in s else "pas patche", GIFT))
        return
    if mode == "revert":
        bak = GIFT + ".pre_qeval"
        if os.path.exists(bak):
            shutil.copyfile(bak, GIFT)
            print("  restaure %s" % GIFT)
        else:
            print("  pas de sauvegarde %s" % bak)
        return
    if MARK in s:
        print("  DEJA PATCHE")
        return

    for i, (old, new) in enumerate(SUBS, 1):
        n = s.count(old)
        if n != 1:
            sys.exit("!! substitution %d : l'ancre apparait %d fois, attendu 1.\n"
                     "   RIEN n'a ete ecrit.\n--- ancre ---\n%s" % (i, n, old))
        s = s.replace(old, new)

    bak = GIFT + ".pre_qeval"
    if not os.path.exists(bak):
        shutil.copyfile(GIFT, bak)
    io.open(GIFT, "w", encoding="utf-8").write(s)
    ast.parse(s)
    print("  patche %s  (sauvegarde .pre_qeval), il compile." % GIFT)
    print("\n  CONTROLE 1 -- rien ne change sans le drapeau :")
    print("    l'en-tete doit dire  Quantile head: aucune -- le CRPS mesure un Dirac")
    print("\n  CONTROLE 2 -- un dataset, deux minutes, avec la tete :")
    print('    PYTHONPATH="$WORK/gift-eval-code/src:$PYTHONPATH" $VENV/bin/python \\')
    print("      example/eval_gift.py --checkpoint runs/R1/step150000.pth \\")
    print("      --json_path checkpoints/reverso_nano/args.json \\")
    print("      --loop-k 4 --loop-beta 0.25 --qhead runs/R1/qhead.pth \\")
    print("      --output-dir /tmp/qtest --dataset m4_weekly")
    print("\n  a lire : 'tete quantile : 9 niveaux sur out_proj', puis un CRPS")
    print("  DIFFERENT de celui de R1 (0.2846). S'il est identique, la tete")
    print("  n'est pas branchee et le Dirac est toujours la.")


if __name__ == "__main__":
    main()
