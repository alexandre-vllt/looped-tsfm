#!/usr/bin/env python3
"""Entraine une tete de regression quantile sur un backbone GELE.

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ

    # 1. TOUJOURS commencer par la : trouver la tete, ne rien entrainer
    $VENV/bin/python train_quantile_head.py --checkpoint runs/R1/step150000.pth --inspect

    # 2. puis entrainer (quelques milliers de steps, < 1 h)
    $VENV/bin/python train_quantile_head.py \\
        --checkpoint runs/R1/step150000.pth --loop-k 4 --loop-beta 0.25 \\
        --out runs/R1/qhead.pth --steps 4000

LA RECETTE EST CELLE DU PAPIER, C.5
  "we repurpose a pretrained Reverso model by training the decoder head with a
   quantile regression objective [...] does not train the full model from
   scratch"
Backbone gele. Leur Table 7 : CRPS 0.459 sur le split medium, meilleur score
du tableau toutes tailles confondues.

CE QUE CA REPARE
example/eval_gift.py fabrique aujourd'hui ses echantillons ainsi :
    samples = np.repeat(pred_full_np[i][None, :], self.num_samples, axis=0)
Cent copies du point. Le CRPS et le mean_weighted_sum_quantile_loss qu'on
rapporte depuis le debut mesurent un Dirac.

LE CONTROLE A LIRE AU STEP 0
La tete quantile est initialisee en recopiant Q fois la tete ponctuelle, donc
au depart tous les quantiles valent la prevision du checkpoint. La perte
pinball initiale doit alors valoir EXACTEMENT la moitie de la MAE du modele
de depart (a q=0.5 la pinball vaut 0.5|e|, et les autres niveaux se
compensent deux a deux sur une erreur symetrique). Le script imprime les deux.
Si elles ne coincident pas a ~1 %, le hook est accroche au mauvais module et
il ne faut PAS entrainer.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reverso_torch.data import make_loader                       # noqa: E402
from reverso_torch.quantile import (DEFAULT_HEAD, DEFAULT_QUANTILES,  # noqa: E402
                                    HeadTap, QuantileHead, find_head,
                                    pinball_loss)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--json_path", default="checkpoints/reverso_nano/args.json")
    p.add_argument("--out", default="")
    p.add_argument("--inspect", action="store_true",
                   help="imprime les candidats pour la tete et sort")
    p.add_argument("--head-name", dest="head_name", default=DEFAULT_HEAD,
                   help="la couche de SORTIE. Par defaut out_proj "
                        "(model.py 380 : Linear(d_model, 1)). Ce n'est PAS "
                        "`head`, qui fabrique les 48 requetes du decodeur.")
    # le modele : doivent valoir CEUX DE L'ENTRAINEMENT du checkpoint
    p.add_argument("--loop-k", dest="loop_k", type=int, default=1)
    p.add_argument("--loop-beta", dest="loop_beta", type=float, default=1.0)
    # les donnees
    _w = os.environ.get("WORK")
    p.add_argument("--data", default=(_w + "/synth") if _w else None)
    p.add_argument("--real", default=(_w + "/real") if _w else None)
    p.add_argument("--p-real", dest="p_real", type=float, default=0.5)
    p.add_argument("--real-alpha", dest="real_alpha", type=float, default=0.5)
    p.add_argument("--sampler", default="resample",
                   choices=["resample", "enumerated"])
    p.add_argument("--no-fla", dest="use_fla", action="store_false",
                   help="chemin torch pur au lieu du noyau fla. A utiliser "
                        "seulement si bfloat16 n'est pas disponible : le noyau "
                        "chunk_delta_rule de fla REFUSE le float32, et l'eval "
                        "passe par fla, donc le desactiver change la fonction "
                        "mesuree.")
    # l'entrainement de la tete
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", dest="log_every", type=int, default=50)
    p.add_argument("--force", action="store_true",
                   help="entrainer malgre un controle de pas 0 en echec. "
                        "N'a de sens que si l'ecart est compris et documente.")
    p.add_argument("--loss-space", dest="loss_space", default="scalefree",
                   choices=["scalefree", "raw"],
                   help="scalefree divise par l'etendue du contexte, comme la "
                        "normalisation min-max interne du modele")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- le modele, charge EXACTEMENT comme a l'evaluation ------------------
    # On reutilise ReversoPredictor plutot que de reconstruire : c'est lui qui
    # sait charger nos checkpoints, preferer les poids EMA et faire le swap
    # fla. Reecrire ce chargement serait l'erreur de eval_sweep.sh (CLAUDE.md 4a).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "example"))
    from eval_gift import ReversoPredictor                        # noqa: E402

    cfg = {}
    if os.path.isfile(a.json_path):
        cfg = json.load(open(a.json_path))
    pred = ReversoPredictor(
        prediction_length=int(cfg.get("output_token_len", 48)),
        checkpoint_path=a.checkpoint, device=str(dev),
        seq_len=int(cfg.get("seq_len", 2048)),
        input_token_len=int(cfg.get("input_token_len", 2048)),
        output_token_len=int(cfg.get("output_token_len", 48)),
        d_model=int(cfg.get("d_model", 128)),
        d_intermediate=int(cfg.get("d_intermediate", 512)),
        output_bottleneck_dim=int(cfg.get("output_bottleneck_dim", 48)),
        expand_v=float(cfg.get("expand_v", 1.0)),
        state_weaving=int(cfg.get("state_weaving", 1)),
        gating_kernel_size=int(cfg.get("gating_kernel_size", 3)),
        main_module=str(cfg.get("main_module",
                                "conv,attn,conv,attn,conv,attn,conv,attn")),
        use_output_pe=bool(cfg.get("use_output_pe", False)),
        batch_size=a.batch_size, use_amp=0,
        loop_k=a.loop_k, loop_beta=a.loop_beta, use_ema=True,
        use_fla=a.use_fla,
    )
    model = pred.model
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    H = int(cfg.get("output_token_len", 48))
    L = int(cfg.get("seq_len", 2048))

    # -- trouver la tete ----------------------------------------------------
    if a.inspect:
        print("\n=== nn.Linear du modele ===")
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                tag = ""
                if n == DEFAULT_HEAD:
                    tag = "  <- LA SORTIE : c'est celle-ci qu'on remplace"
                elif m.out_features == H:
                    tag = "  <- les requetes du decodeur, PAS la sortie"
                print(f"  {n:<55} {m.in_features:>6} -> {m.out_features:<6}{tag}")
        print(f"""
  model.py 376-405 :
      self.head     = nn.Linear(input_token_len, output_bottleneck_dim)   2048 -> 48
      self.out_proj = nn.Linear(d_model, 1)                                 32 -> 1

      temp_out = self.head(dec_out).permute(0, 2, 1)
      q = self.simple_q_proj(temp_out)      # les 48 REQUETES
      ...
      dec_out = self.out_proj(attn)         # la SORTIE

  `head` produit les positions de requete du decodeur, pas la prevision --
  malgre son nom et malgre ses 48 canaux. La couche a remplacer est
  {DEFAULT_HEAD}, qui donne une valeur par position.""")
        return

    name, head = find_head(model, a.head_name)
    print(f"  couche de sortie : {name}  "
          f"({head.in_features} -> {head.out_features})")

    qs = DEFAULT_QUANTILES
    qhead = QuantileHead(head, len(qs)).to(dev)
    n_par = sum(q.numel() for q in qhead.parameters())
    print(f"  tete quantile : {len(qs)} niveaux, {n_par:,} parametres entraines")
    print(f"  backbone gele : {sum(q.numel() for q in model.parameters()):,} "
          f"parametres figes")

    # BFLOAT16 OBLIGATOIRE AVEC fla.
    # chunk_delta_rule leve "does not support float32. Please use bfloat16."
    # example/eval_gift.py enveloppe son forward dans autocast bfloat16 ; on
    # fait pareil, ce qui a l'avantage de mesurer EXACTEMENT la fonction que
    # l'evaluation mesure. La tete quantile reste en float32 : 297 parametres,
    # aucune raison de les entrainer en demi-precision.
    use_bf16 = (dev.type == "cuda" and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported())
    if a.use_fla and not use_bf16:
        sys.exit("!! fla exige bfloat16 et ce GPU ne le supporte pas. "
                 "Relancer avec --no-fla (mais la fonction mesuree differera "
                 "alors de celle de l'evaluation).")
    print(f"  backbone en {'bfloat16 (autocast)' if use_bf16 else 'float32'}, "
          f"tete en float32")

    opt = torch.optim.AdamW(qhead.parameters(), lr=a.lr, weight_decay=0.0)
    dl = make_loader(data_dir=a.data, real_dir=a.real, seq_len=L, horizon=H,
                     batch_size=a.batch_size, seed=a.seed, p_real=a.p_real,
                     num_workers=a.workers, verify="quick",
                     real_alpha=a.real_alpha, sampler=a.sampler)

    print(f"\n  {'step':>6} {'pinball':>10} {'0.5*MAE pt':>11} {'MAE med':>9} "
          f"{'MAE pt':>9}")
    t0 = time.time()
    step = 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        xm = torch.zeros_like(x)
        ym = torch.zeros(x.shape[0], H, x.shape[-1], device=dev, dtype=x.dtype)

        with torch.no_grad(), HeadTap(model, name) as tap:
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    point = model(x, xm, ym)              # (B, ?, C)
            else:
                point = model(x, xm, ym)
        feats = tap.value
        if feats is None:
            sys.exit("!! le hook n'a rien capture -- mauvais module ?")

        point = point[:, -H:, :].squeeze(-1).to(torch.float32)   # (B, H)
        # feats est l'entree de out_proj : (B, 48, d_model). La tete quantile
        # est Linear(d_model, Q), donc qhead(feats) donne (B, 48, Q)
        # directement -- aucun reshape.
        yt = y.squeeze(-1)                                # (B, H), BRUT

        # model.py 384-407 : le modele normalise x en min-max, et APRES
        # out_proj refait dec_out * x_range + x_min. `feats` etant l'entree de
        # out_proj, qhead(feats) vit dans l'espace NORMALISE -- pas celui de y.
        # Il faut donc le ramener dans les unites de y avec EXACTEMENT les
        # memes statistiques, clamp 1e-5 compris (model.py 388).
        xs = x.squeeze(-1)
        x_min_b = xs.amin(1)                                     # (B,)
        x_range_b = (xs.amax(1) - x_min_b).clamp_min(1e-5)       # (B,)
        scale = x_range_b if a.loss_space == "scalefree" else None

        qpred = qhead(feats.to(torch.float32))            # (B, H, Q) normalise
        qpred = qpred * x_range_b.view(-1, 1, 1) + x_min_b.view(-1, 1, 1)
        loss = pinball_loss(qpred, yt.to(torch.float32), qs, scale)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qhead.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == 0:
            with torch.no_grad():
                med = qpred[..., len(qs) // 2]
                d = (yt - point).abs()
                dm = (yt - med).abs()
                if scale is not None:
                    d = d / scale.view(-1, 1); dm = dm / scale.view(-1, 1)
                mae_pt, mae_med = d.mean().item(), dm.mean().item()
            print(f"  {step:>6} {loss.item():>10.5f} {0.5 * mae_pt:>11.5f} "
                  f"{mae_med:>9.5f} {mae_pt:>9.5f}")
            if step == 0:
                r = loss.item() / max(0.5 * mae_pt, 1e-12)
                print(f"\n  CONTROLE step 0 : pinball / (0.5 x MAE point) = {r:.4f}")
                if abs(r - 1.0) > 0.02:
                    print("  !! ARRET : attendu 1.000. La tete quantile part d'une")
                    print("     copie exacte de la tete ponctuelle, donc les deux")
                    print("     doivent coincider. Un ecart signifie que le hook")
                    print("     n'est pas sur la bonne couche, que `point` n'est")
                    print("     pas la sortie de cette tete, ou que les deux ne")
                    print("     sont pas dans le MEME ESPACE (model.py 405-407 :")
                    print("     dec_out = dec_out * x_range + x_min APRES out_proj).")
                    print("     Le 2 novembre ce controle a vu 1.27e7 et le script")
                    print("     a quand meme entraine 4000 steps pour rien.")
                    print("     Il s'arrete desormais. --force pour passer outre,")
                    print("     et dans ce cas ecrire pourquoi dans done.md.\n")
                    if not a.force:
                        raise SystemExit(3)
                else:
                    print("  ok : la tete reproduit la tete ponctuelle a "
                          f"{abs(r - 1.0) * 100:.3f} % pres.\n")

        step += 1
        if step >= a.steps:
            break

    print(f"\n  {step} steps en {time.time() - t0:.0f}s")
    if a.out:
        torch.save({"qhead": qhead.state_dict(), "quantiles": list(qs),
                    "head_name": name, "src_checkpoint": a.checkpoint,
                    "loop_k": a.loop_k, "loop_beta": a.loop_beta,
                    "loss_space": a.loss_space, "steps": step}, a.out)
        print(f"  ecrit {a.out}")
        print("\n  Reste a brancher cette tete dans example/eval_gift.py :")
        print("  remplacer le np.repeat par les quantiles, et REINJECTER LA")
        print("  MEDIANE dans le rollout autoregressif (pas la moyenne des")
        print("  quantiles) pour que la MASE reste comparable aux autres bras.")


if __name__ == "__main__":
    main()
