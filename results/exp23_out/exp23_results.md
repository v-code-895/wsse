# Experiment 23: an exception unit vs head fine-tunes (MobileNetV2, AFHQ)

Base: {"fit": {"cat_acc": 0.925, "dog_acc": 0.97, "wild_called_cat": 0.0}, "val": {"cat_acc": 0.954, "dog_acc": 0.954, "wild_called_cat": 0.0}}
Validation cat errors by predicted class: {"lynx": 19, "leopard": 1, "Angora": 3}
Patch: {"theta": 0.12475285723748625, "beta": 7.365071100747103, "fit_errors": 45, "fit_errors_where_unit_fires": 45, "new_params": 1287, "fit_seconds": 0.35324573516845703}

| fix | cat acc | errors fixed | correct cats broken | dogs changed | dogs → cat | wild changed | wild → cat | KL dogs | KL wild | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.954 | 0/23 | 0 | 0 | 0 | 0 | 0 | 0.00000 | 0.00000 | 0.00 |
| exception unit | 0.998 | 22/23 | 0 | 0 | 0 | 1 | 1 | 0.00009 | 0.00039 | 0.35 |
| head fine-tune, with preservation | 0.998 | 22/23 | 0 | 2 | 0 | 2 | 1 | 0.00074 | 0.00166 | 3.04 |
| head fine-tune, errors only | 1.000 | 23/23 | 0 | 377 | 375 | 338 | 336 | 2.27116 | 2.23698 | 1.88 |

Exception unit fired on: {"cat": 499, "dog": 73, "wild": 33}; max logit change where it is off: 0.0
