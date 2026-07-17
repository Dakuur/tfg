# Predicció de Metàstasi en Càncer Colorectal amb Whole Slide Images usant Graph Attention Networks

Treball de Fi de Grau — Enginyeria de Dades, Escola d'Enginyeria (EE), Universitat Autònoma de Barcelona (UAB)
Curs 2025/26, desenvolupat al [Computer Vision Center (CVC)](https://www.cvc.uab.es/)

**Autor:** David Morillo Massagué
**Tutora:** Dra. Debora Gil Resina (DCC)

## Resum

Es presenta un sistema de classificació N0/N1 per a la detecció de metàstasi en ganglis limfàtics en càncer colorectal pT1 a partir de *Whole Slide Images*. El teixit es representa com un graf espacial sobre les coordenades dels *patches*, de manera que es preserva la topologia real del teixit. Cada node és un *patch* de 2048×2048 px representat per l'embedding CLS (vector de forma `[1, 1536]`) del model UNI2. Es comparen dues arquitectures configurables: un GAT de referència amb *readout* pla (mean, max, mean_max o attention) i un GAT amb DiffPool jeràrquic intercalat per obtenir representacions a múltiples escales. El diagnòstic per pacient s'obté per agregació MIL sobre les seccions histològiques. La cerca d'hiperparàmetres combina dos estudis Optuna (5-fold CV + retrain 90/10 de les millors configuracions obtingudes). El millor model és el GAT de referència, que obté AUC test = 0,875, sensibilitat 100% i especificitat 57,5%.

## Abstract

We present an N0/N1 classification system for lymph node metastasis prediction in pT1 colorectal cancer from Whole Slide Images. Tissue is represented as a spatial graph over patch coordinates, preserving the real spatial topology of the tissue. Each node is a 2048×2048 px patch represented by a CLS embedding of shape `[1, 1536]` from the UNI2 model. Two configurable architectures are compared: a GAT baseline with flat readout (mean, max, mean_max or attention) and a GAT+DiffPool model with hierarchical DiffPool interleaved between GAT layers for multi-scale representations. Patient-level diagnosis is obtained via MIL aggregation over histological sections. Hyperparameter search combines two Optuna studies (5-fold CV + 90/10 retraining of the best configurations). The best model is the GAT baseline, achieving test AUC = 0.875, sensitivity 100% and specificity 57.5%.

## Paraules clau / Keywords

Càncer colorectal pT1, Metàstasi ganglionar, Histopatologia digital, Graph Attention Networks, DiffPool jeràrquic, Graf espacial de Delaunay, MIL

*pT1 colorectal cancer, Lymph node metastasis, Digital histopathology, Graph Attention Networks, Hierarchical DiffPool, Spatial Delaunay graph, Multiple Instance Learning*

## Estructura del repositori

- [`article/`](article/) — memòria del TFG en LaTeX ([PDF](article/TFG.pdf))
- [`scripts/`](scripts/) — pipeline de preprocessament, construcció de grafs i entrenament
- [`frontend/`](frontend/) — interfície web de visualització i inferència
- [`pt1diagnosis/`](pt1diagnosis/) — submòdul amb el model i la cerca d'hiperparàmetres, desenvolupat conjuntament amb la resta de l'equip ([repositori](https://github.com/IAM-CVC/PT1Diagnosis))
- [`Code-IAM-Server/`](Code-IAM-Server/) — codi del servidor del CVC

## Contacte

David Morillo Massagué — [David.MorilloMa@autonoma.cat](mailto:David.MorilloMa@autonoma.cat)
