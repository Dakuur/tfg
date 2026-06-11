# FINALIZE_TODO — Substitucions pendents al TFG.tex

> **Quan**: després d'executar `python -m sweep --finalize` + `python -m sweep --report`
> **Fonts dels números** (3 fitxers a `~/outputs/sweep/`):
> - `report.md` / `report.json` — sweep complet, importància fANOVA, top trials, breakdowns
> - `final_results.json` — mètriques al test (Sens/Spec/AUC/threshold + per hospital)
> - `best_params_baseline.json` / `best_params_diffpool.json` — best hiperparàmetres per study
>
> **També útil**:
> - `python -m sweep --status` per al recompte de trials per arquitectura
> - `progress.csv` per a top N trials

---

## Convenció

- `X` = placeholder. Substituir per el valor numèric (usar coma decimal: `0,512`)
- "Baseline" i "DiffPool" són els 2 studies aïllats
- "guanyador" = el study amb major `objective` CV (consultar `best_params.json` global)

---

## §8 Resultats — cos del TFG

### Línies 607-609: volum d'experiments

```latex
$N_{\text{base}}=X$ trials per a GAT Baseline
$N_{\text{diff}}=X$ per a GAT+DiffPool
$N_{\text{tot}}=X$
```

Font: `report.json["per_arch_best"]` o `python -m sweep --status` (suma `complets + prunats`).

---

### Línies 627-631: Taula `tab:winners` — millor per arquitectura

Una fila per arquitectura, **8 columnes**:

| Columna | Font | Notes |
|---|---|---|
| `N_trials` | `python -m sweep --status` | Trials totals (complets + prunats) per study |
| `Pooling` | `report.json["per_arch_best"][arch]["params"]["pooling"]` | Per Baseline. Per DiffPool: sempre `diff`, pots posar el `diff_final_pool` |
| `MIL` | `["params"]["mil"]` | mean / max / noisy_or / attention |
| `Obj. CV` | `["user_attrs"]["spec_mean"]` − 10·max(0, 1−sens_mean) | Pots posar el `value` del best trial directament |
| `AUC CV` | `["user_attrs"]["auc_mean"]` ± std | std no està guardat directament; cal calcular-lo dels fold metrics |
| `AUC test` | `final_results.json["test"]["auc"]` | **NOMÉS després del --finalize** |
| `Sens / Espec test` | `final_results.json["test"]["at_threshold"]["sens" / "spec"]` | Al llindar $t^*$ mediana |

**IMPORTANT**: només el guanyador té `final_results.json` (--finalize entrena un sol model). Per a l'altre study, posa `—` o fes `--finalize --arch <l'altre>` per tenir ambdós.

---

### Línies 651, 660-661: Taula `tab:cm` — matriu confusió

```latex
sensibilitat = X, especificitat = X, VPP = X, VPN = X
              N0 reals   N1 reals
Predicció N0    X          X      ← TN, FN
Predicció N1    X          X      ← FP, TP
```

Font: `final_results.json["test"]["at_threshold"]`:
- `tp`, `fn`, `tn`, `fp` (al diccionari de mètriques)
- `sens`, `spec`, `ppv`, `npv`

---

### Línia 677: paràgraf "Resum eixos més influents"

```latex
Els eixos més influents observats són $X$ per a Baseline i $X$ per a DiffPool.
```

Font: `report.json["param_importances"][arch]` → ordenar per valor i agafar els 2-3 primers.

---

### Línies 721-722: Taula `tab:comparison` — GAT vs guies/SOTA

Files **GAT (llindar 0.5)** i **GAT (llindar $t^*$)**. **8 valors a omplir** (4 + 4):

| Columna | Font |
|---|---|
| `Sens.` | `final_results.json["test"]["at_0.5"]["sens"]` i `["at_threshold"]["sens"]` |
| `Espec.` | `["at_0.5"]["spec"]` i `["at_threshold"]["spec"]` |
| `AUC` | `["auc"]` (mateix per a les 2 files: l'AUC no depèn del llindar) |
| `Bal.Acc.` | `(sens + spec) / 2` calculat |
| `t^*` (a la 2a fila) | `final_results.json["threshold_final"]` |

---

### Línia 740: caption de `fig:roc-top5`

```latex
El cercle gros marca el llindar $t^*=X$ derivat exclusivament del val set
```

Font: `final_results.json["threshold_final"]`.

**Acció addicional**: regenerar `img/roc_top5.pdf` amb les corbes dels millors de cada arquitectura (no top-5 del grid antic). Script suggerit:

```python
import json, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Carregar probs+labels del best baseline i diffpool (folds CV)
# Fer roc_curve per cada un, dibuixar al pdf
# Afegir punts de JSCCR, NCCN, ESMO, LightGBM com a marcadors
# Guardar a img/roc_top5.pdf
```

---

### Línies 757, 760-762: impacte clínic

```latex
\item GAT (llindar $t^*$): $X$ cirurgies innecessàries.
... redueix en $\sim X$ pacients respecte JSCCR i $\sim X$ respecte NCCN/ESMO.
Sobre els 40 N0 del test, això són $X$ FP en lloc dels $\sim 32$
```

Càlcul a partir de `final_results.json["test"]["at_threshold"]["spec"]`:
- `(1 - spec) * 100` = cirurgies innecessàries per 100 N0
- `81 - aquest_valor` = reducció respecte JSCCR
- `49 - aquest_valor` ≈ reducció respecte NCCN/ESMO (mitjana)
- `(1 - spec) * 40` = FP absoluts sobre els 40 N0 del test

---

### Línies 791-792: Conclusions

```latex
el model X obté la millor combinació de sensibilitat i especificitat al test
(AUC $X$, Sens.\ $X$, Espec.\ $X$)
```

Substituir:
- `el model X` → `el model Baseline` o `el model DiffPool` (segons guanyador)
- 3 valors: AUC test, Sens test @ $t^*$, Spec test @ $t^*$ (del guanyador)

---

### Línia 802: paràgraf "Sobre el conjunt de dades"

```latex
$X$ supera $X$ en $X$ punts d'AUC al test [per omplir]
```

Substituir:
- `$X$ supera $X$` → `Baseline supera DiffPool` (o a l'inrevés)
- `$X$ punts` → diferència absoluta d'AUC test × 100

També **revisar el text del paràgraf**: l'argument actual ("seccions petites → DiffPool sub-òptim") és per si Baseline guanya. **Si guanya DiffPool, cal canviar el to** a "el pooling jeràrquic compensa la grandària reduïda de les seccions" o similar.

---

### Línies 814-815, 820-822, 830-834, 840: §10 Discussió Comparativa

Substituir tots els `$X$` per:
- AUC test del guanyador → línia 814
- Espec. (al $t^*$) → línia 815
- VPP al test (prevalença 28.6%) → línia 820
- VPP a prevalença 10% → línia 822 (calcular per Bayes: $\text{VPP}_{p=0.1} = \frac{0.1 \cdot \text{sens}}{0.1 \cdot \text{sens} + 0.9 \cdot (1-\text{spec})}$)
- VPN a prevalença 10% → mateixa línia (Bayes)
- Sens. @ 0.5 → línia 830
- $t^*$ → línia 832
- Sens / Spec / millora sobre JSCCR (en punts) → línies 833-834
- Espec. al mateix punt operatiu (vs LightGBM 85.8%) → línia 840

---

### Línia 850: Limitacions

```latex
l'AUC CV $X \pm X$ sobre cinc folds
```

Substituir per AUC CV mean ± std del guanyador. Calcular l'std dels `fold_metrics` del trial al `metrics.json` del trial dir.

---

### Línies 865-866: Limitació 4 (llindar)

```latex
folds com a $t^*=X$. Aplicat al test, dona Sens. $X$ amb Espec. $X$.
```

Substituir:
- `t^*` → `final_results.json["threshold_final"]`
- Sens / Spec @ $t^*$ → `["test"]["at_threshold"]["sens" / "spec"]`

---

## Annex A.2 — Taula `tab:main-effects` (importància fANOVA)

### Caption (línies 984-985):

```latex
Baseline: $N_{\text{base}}=X$ trials. DiffPool: $N_{\text{diff}}=X$.
```

Mateixos valors que la §8.

### Cos de la taula (línies 994-1012)

**Per cada eix, dos valors d'importància** (0-1). Font:

```python
# Al report.json:
report.json["param_importances"]["baseline"]    # dict {eix: valor}
report.json["param_importances"]["diffpool"]    # dict {eix: valor}
```

Eixos a omplir:

**Específics**:
- `pooling`, `n_gat_layers` → només Baseline (DiffPool: `---`)
- `n_diffpool_layers`, `diff_K_top`, `diff_K_bottom`, `aux_loss_weight`, `diff_final_pool` → només DiffPool (Baseline: `---`)

**Compartits** (2 valors per fila):
- `hidden`, `heads`, `dropout`, `mil`, `lr`, `weight_decay`, `pos_weight`, `optimizer`, `scheduler`, `grad_clip`

> **Nota**: si fANOVA falla per algun eix (massa pocs trials o eix degenerat), Optuna fa fallback a MeanDecreaseImpurity. El valor encara és vàlid però mencionar-ho a la caption si afecta.

---

## Annex A.3 — Taula `tab:best-params` (millors hiperparàmetres)

Llegir des de `best_params_baseline.json["params"]` i `best_params_diffpool.json["params"]`.

| Línia | Camp baseline | Camp diffpool |
|---|---|---|
| 1038 | `params["pooling"]` | (sempre `diff`, ja escrit) |
| 1039 | `params["n_gat_layers"]` | `params["n_diffpool_layers"] + 1` |
| 1040 | --- | `params["n_diffpool_layers"]` |
| 1041 | --- | `params["diff_K_top"]`, `params["diff_K_bottom"]` |
| 1042 | --- | `params["aux_loss_weight"]` |
| 1043 | --- | `params["diff_final_pool"]` |
| 1046 | `params["hidden"]`, `params["heads"]`, `params["dropout"]` | mateixos camps diffpool |
| 1047 | `params["mil"]` | `params["mil"]` |
| 1048 | `params["lr"]`, `params["weight_decay"]`, `params["pos_weight"]` | mateixos camps diffpool |
| 1049 | `params["optimizer"]`, `params["scheduler"]` | mateixos camps diffpool |
| 1052 | `user_attrs["spec_mean"]−10·max(0,1−sens_mean)`, `["auc_mean"]`, `["sens_mean"]`, `["spec_mean"]` | mateixos camps diffpool |

---

## Després de substituir totes les X

1. **Recompilar**:
   ```bash
   cd article && pdflatex TFG.tex && biber TFG && pdflatex TFG.tex && pdflatex TFG.tex
   ```
2. **Verificar que no queda cap `X` no substituïda**:
   ```bash
   grep -nE '\s X\s|\$X\$|\$X\b|= X\b|\\!=\\!X' TFG.tex
   ```
   (Hauria de retornar 0 línies)
3. **Comptar pàgines** del cos (abans de `\appendix`): han de ser ≤ 10
4. **Regenerar `img/roc_top5.pdf`** amb les corbes ROC dels 2 best (Baseline + DiffPool) sobre el test
5. **Revisar el text dels paràgrafs `[per omplir]` i les conclusions** — si DiffPool guanya, cal invertir alguns arguments

---

## Comprovacions de coherència

Després d'omplir:
- **Sensibilitat al $t^*$** = 1.0 (o molt a prop) → si no, el threshold no es va fixar bé
- **AUC test** dins del rang `AUC CV mean ± 2·std` → si fora, possible overfitting
- **L'arquitectura del millor params** ha de coincidir amb la del guanyador a `tab:winners`
- **VPP a prevalença 10%** sempre < VPP al test (perquè la prevalença del test és superior)
