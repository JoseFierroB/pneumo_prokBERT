#!/usr/bin/env python3
"""Model builders: thin stable wrappers over scripts.train_mix223k pieces."""
import sys
sys.path.insert(0, ".")


def build_model(args, device):
    """Apply dropout/freeze/head/optimizer options. Returns (model, info dict)."""
    from scripts import train_mix223k as T

    do_head, do_hidden, do_attn = T.resolve_dropout(args)
    print(f"Dropout blocks: head={do_head} hidden={do_hidden} attn={do_attn}")
    model = T.load_model(device, dropout=args.dropout, dropout_head=do_head,
                         dropout_hidden=do_hidden, dropout_attn=do_attn)
    if args.probe_checkpoint and not (args.freeze_backbone or args.freeze_bottom_k is not None
                                      or args.freeze_top_k is not None):
        raise SystemExit("--probe-checkpoint requires --freeze-backbone/--freeze-bottom-k/--freeze-top-k")
    if sum([bool(args.freeze_backbone), args.freeze_bottom_k is not None,
            args.freeze_top_k is not None]) > 1:
        raise SystemExit("--freeze-backbone/--freeze-bottom-k/--freeze-top-k are mutually incompatible")
    if args.probe_checkpoint:
        T.load_probe_backbone_(model, args.probe_checkpoint)
    if args.mlp_head:
        T.apply_mlp_head_(model, p=do_head)
    if args.freeze_backbone:
        T.freeze_backbone_(model)
    if args.freeze_bottom_k is not None:
        T.freeze_bottom_k_(model, args.freeze_bottom_k)
    if args.freeze_top_k is not None:
        T.freeze_top_k_(model, args.freeze_top_k)
    layer_profile = None
    if args.dropout_layers:
        layer_profile = T.apply_layer_dropout_(model, T.parse_layer_profile(args.dropout_layers))
        print(f"Layer dropout profile: {[round(p, 4) for p in layer_profile]}")
    return model, {"dropout_head": do_head, "dropout_hidden": do_hidden,
                   "dropout_attn": do_attn, "dropout_layers": layer_profile}
