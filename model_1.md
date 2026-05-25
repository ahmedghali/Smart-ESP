# Modèle 1 — LSTM Predictor (BiLSTM + Multi-Head Attention)

> **Rôle** : Lire les **168 dernières heures** des 11 capteurs ESP et **prédire** si une panne va survenir dans les **48 prochaines heures**.
> **Architecture finale (finetuned)** : BiLSTM compact (hidden=32, 1 couche) + 4 têtes d'attention + tête de reconstruction (multi-task).
> **Résultat** : **AUC = 0.6914** sur données réelles Z1/Z2/Z3.

---

## 1. Vue d'ensemble du flux de données

```
ENTRÉE                              SORTIE
┌───────────────────┐               ┌──────────────────────────────┐
│ Séquence de 168 h │               │ ŷ ∈ [0, 1]                   │
│ × 11 capteurs     │ ─── MODÈLE ──▶│ "Probabilité de panne        │
│ (7 jours d'hist.) │               │  dans les 48 h à venir"      │
└───────────────────┘               │                              │
                                    │ + x̂ ∈ ℝ¹¹ (reconstruction,  │
                                    │   utilisé uniquement pendant │
                                    │   l'entraînement)            │
                                    └──────────────────────────────┘
```

**Tenseur d'entrée** : `X ∈ ℝ^(B × 168 × 11)` où :
- `B` = taille de batch (typiquement 64)
- `168` = longueur de la fenêtre temporelle (heures)
- `11` = nombre de capteurs DHM

**Tenseur de sortie** :
- `ŷ ∈ ℝ^(B × 1)` : logit de classification (panne / pas panne)
- `x̂ ∈ ℝ^(B × 168 × 11)` : reconstruction de l'entrée (auxiliaire)

---

## 2. Architecture détaillée — bloc par bloc

Le modèle est un **pipeline de 6 étapes**. À chaque étape, le tenseur change de forme.

### 📐 Tableau récapitulatif des formes (pour le schéma)

| # | Bloc | Entrée | Sortie | Paramètres |
|---|------|--------|--------|------------|
| 0 | Input | — | `(B, 168, 11)` | 0 |
| 1 | Input Projection (Linear) | `(B, 168, 11)` | `(B, 168, 32)` | 11×32 + 32 = **384** |
| 2 | BiLSTM (1 couche) | `(B, 168, 32)` | `(B, 168, 64)` | ~**16 896** |
| 3 | LayerNorm | `(B, 168, 64)` | `(B, 168, 64)` | 2×64 = **128** |
| 4 | Multi-Head Attention (4 têtes) | `(B, 168, 64)` | `(B, 168, 64)` | 4×(64²+64) ≈ **16 640** |
| 5 | Residual + Global Avg Pooling | `(B, 168, 64)` | `(B, 64)` | 0 |
| 6a | Classifier head (MLP) | `(B, 64)` | `(B, 1)` | ~**2 600** |
| 6b | Reconstruction head | `(B, 168, 64)` | `(B, 168, 11)` | ~**400** |
| | **TOTAL** | | | **~36 000** |

---

## 3. Détail mathématique de chaque bloc

### 🔵 Bloc 1 — Input Projection

Une couche linéaire qui projette les 11 capteurs bruts vers l'espace caché de dimension 32.

$$\mathbf{x}'_t = \mathbf{W}_{in} \mathbf{x}_t + \mathbf{b}_{in}, \quad \mathbf{W}_{in} \in \mathbb{R}^{32 \times 11}$$

**Pourquoi ?** Élève la dimension pour que le LSTM ait plus de capacité expressive sans changer le nombre de capteurs.

---

### 🟦 Bloc 2 — Bidirectional LSTM (cœur du modèle)

Un LSTM **bidirectionnel** = deux LSTM qui lisent la séquence en parallèle :
- **Forward** ($\overrightarrow{LSTM}$) : lit de $t=1$ à $t=168$
- **Backward** ($\overleftarrow{LSTM}$) : lit de $t=168$ à $t=1$

Pour chaque pas de temps $t$, les deux sorties sont **concaténées** :

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{64}$$

**Équations d'une cellule LSTM** (à chaque pas $t$) :
$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{x}'_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \quad \text{(forget gate)}\\
\mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{x}'_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \quad \text{(input gate)}\\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{x}'_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \quad \text{(output gate)}\\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c \mathbf{x}'_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c) \quad \text{(candidate)}\\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell state)}\\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden state)}
\end{aligned}
$$

**Configuration** :
- Taille cachée par direction : **32**
- Nombre de couches : **1** (un seul stack)
- Bidirectionnel : **oui** → sortie totale = 32 + 32 = **64**

**Pourquoi compact ?** Le modèle initial avait hidden=256 (5.4M paramètres) → **overfit** sur les 76 événements de panne disponibles. Réduction à hidden=32 (36k params) → AUC monte de 0.50 → 0.62.

---

### 🟧 Bloc 3 — Layer Normalization

Normalise chaque vecteur $\mathbf{h}_t$ sur sa dimension :

$$\hat{\mathbf{h}}_t = \frac{\mathbf{h}_t - \mu_t}{\sqrt{\sigma_t^2 + \varepsilon}} \cdot \gamma + \beta$$

où $\mu_t, \sigma_t$ sont la moyenne/écart-type calculés sur les 64 composantes de $\mathbf{h}_t$, et $\gamma, \beta \in \mathbb{R}^{64}$ sont apprenables.

**Pourquoi ?** Stabilise l'entraînement et accélère la convergence.

---

### 🟩 Bloc 4 — Multi-Head Self-Attention (4 têtes)

C'est ici que le modèle **apprend quels pas de temps sont importants** pour la prédiction.

#### Étape 1 : projections Q, K, V

Pour chaque tête $i \in \{1, 2, 3, 4\}$ :
- **Query** : $\mathbf{Q}_i = \hat{\mathbf{H}} \mathbf{W}_i^Q$
- **Key**   : $\mathbf{K}_i = \hat{\mathbf{H}} \mathbf{W}_i^K$
- **Value** : $\mathbf{V}_i = \hat{\mathbf{H}} \mathbf{W}_i^V$

Chaque tête travaille en dimension $d_k = 64 / 4 = 16$.

#### Étape 2 : produit scalaire mis à l'échelle (scaled dot-product)

$$\text{Attention}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_k}}\right) \mathbf{V}_i$$

Le tenseur de **poids d'attention** $\alpha \in \mathbb{R}^{168 \times 168}$ indique à quel point chaque pas de temps "regarde" chaque autre pas de temps. **Ces poids sont exposés pour l'explicabilité (XAI).**

#### Étape 3 : concaténation des 4 têtes

$$\text{MultiHead} = [\text{Attn}_1 \,\|\, \text{Attn}_2 \,\|\, \text{Attn}_3 \,\|\, \text{Attn}_4] \mathbf{W}^O$$

Forme finale : `(B, 168, 64)`.

---

### 🟪 Bloc 5 — Residual Connection + Global Average Pooling

#### Connexion résiduelle (skip connection)

$$\mathbf{R}_t = \hat{\mathbf{h}}_t + \text{MultiHead}_t$$

**Pourquoi ?** Préserve l'information temporelle brute du LSTM si l'attention l'efface.

#### Global Average Pooling sur la dimension temporelle

$$\bar{\mathbf{r}} = \frac{1}{168} \sum_{t=1}^{168} \mathbf{R}_t \in \mathbb{R}^{64}$$

**Effet** : on transforme une séquence de 168 vecteurs en **un seul vecteur résumé** de dimension 64 qui capture toute l'histoire.

---

### 🟨 Bloc 6a — Classifier Head (MLP)

Petit perceptron à 3 couches qui transforme le vecteur résumé en logit :

$$
\bar{\mathbf{r}} \xrightarrow{\text{Linear}(64 \to 32)} \xrightarrow{\text{ReLU + Dropout}} \xrightarrow{\text{Linear}(32 \to 16)} \xrightarrow{\text{ReLU + Dropout}} \xrightarrow{\text{Linear}(16 \to 1)} \text{logit } z
$$

La **probabilité de panne** est obtenue par sigmoïde :

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]$$

---

### 🟫 Bloc 6b — Reconstruction Head (régularisateur)

**Présent uniquement dans la version finetuned** (wrapper multi-task).

Décodeur qui prend la séquence enrichie après attention et tente de **reconstruire l'entrée originale** :

$$
\mathbf{R} \in \mathbb{R}^{168 \times 64} \xrightarrow{\text{Linear}(64 \to 32) + \text{ReLU}} \xrightarrow{\text{Linear}(32 \to 11)} \hat{\mathbf{X}} \in \mathbb{R}^{168 \times 11}
$$

**Pourquoi ?** Sans cette tête, l'encodeur dégénère (AUC=0.50). La reconstruction force le LSTM à apprendre des features **utiles et générales**, pas seulement le shortcut "deviner la classe majoritaire".

---

## 4. Fonction de coût multi-tâche

$$\boxed{\mathcal{L} = \alpha_{cls} \cdot \mathcal{L}_{focal}(\hat{y}, y) + (1 - \alpha_{cls}) \cdot \mathcal{L}_{MSE}(\hat{\mathbf{X}}, \mathbf{X})}$$

avec $\alpha_{cls} = 0{,}7$ (70% classification, 30% reconstruction).

### Focal Loss (pour le déséquilibre de classes)

$$\mathcal{L}_{focal} = -\text{pw} \cdot (1 - \hat{y})^\gamma \cdot y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})$$

avec $\gamma = 2$ et $\text{pw} \approx 6{,}9$ (poids géométrique post-augmentation).

### MSE Reconstruction

$$\mathcal{L}_{MSE} = \frac{1}{B \cdot 168 \cdot 11} \sum_{b,t,c} (x_{b,t,c} - \hat{x}_{b,t,c})^2$$

---

## 5. Pipeline d'entraînement en 2 étapes

```
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 — Pré-entraînement sur données synthétiques       │
│  ────────────────────────────────────────────────          │
│  • 50 000 lignes générées par SyntheticESPDataGenerator     │
│  • Modèle large : hidden=256, 3 couches, 8 têtes (5.4M)    │
│  • Loss : BCE simple                                        │
│  • Objectif : apprendre la physique générale d'une panne   │
│  • Résultat : Accuracy=87.3%, AUC=0.62 sur synthétique     │
└─────────────────────────────────────────────────────────────┘
                              ↓
                      Transfert des poids
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 — Fine-tuning sur données réelles                 │
│  ────────────────────────────────────────────────          │
│  • 3 puits algériens Z1, Z2, Z3 (296 393 lignes)           │
│  • Split intra-well : 70% train / 30% test par puits       │
│  • Modèle COMPACT : hidden=32, 1 couche, 4 têtes (36k)     │
│  • Wrapper MultiTask : ajout de la tête de reconstruction  │
│  • Loss : 0.7·FocalLoss + 0.3·MSE                          │
│  • Augmentation : jitter σ=0.08 sur séquences de panne     │
│  • Early stopping : epoch 5 (convergence en 2 epochs)      │
│  • Résultat : AUC=0.6914 sur données réelles ✅            │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Schéma d'architecture (pour le dessin)

```
                  ┌─────────────────────────────────────────────┐
                  │  ENTRÉE                                     │
                  │  X ∈ ℝ^(B × 168 × 11)                       │
                  │  168 h × 11 capteurs DHM                    │
                  └─────────────────────┬───────────────────────┘
                                        │
                  ┌─────────────────────▼───────────────────────┐
                  │  BLOC 1 — Input Projection                  │
                  │  Linear(11 → 32)                            │
                  │  Sortie : (B, 168, 32)                      │
                  └─────────────────────┬───────────────────────┘
                                        │
                  ┌─────────────────────▼───────────────────────┐
                  │  BLOC 2 — Bidirectional LSTM                │
                  │   ┌───────────────┐  ┌───────────────┐      │
                  │   │ Forward LSTM  │  │ Backward LSTM │      │
                  │   │ hidden=32     │  │ hidden=32     │      │
                  │   └───────┬───────┘  └───────┬───────┘      │
                  │           └────────┬─────────┘              │
                  │           Concaténation [→ + ←]             │
                  │           Sortie : (B, 168, 64)             │
                  └─────────────────────┬───────────────────────┘
                                        │
                  ┌─────────────────────▼───────────────────────┐
                  │  BLOC 3 — Layer Normalization               │
                  │  Sortie : (B, 168, 64)                      │
                  └─────────────────────┬───────────────────────┘
                                        │
                  ┌─────────────────────▼───────────────────────┐
                  │  BLOC 4 — Multi-Head Self-Attention         │
                  │           ┌─ Tête 1 (d_k=16) ─┐             │
                  │   QKV ───▶├─ Tête 2 (d_k=16) ─┤── Concat    │
                  │           ├─ Tête 3 (d_k=16) ─┤   + W^O     │
                  │           └─ Tête 4 (d_k=16) ─┘             │
                  │   Poids d'attention α ∈ ℝ^(168×168)        │
                  │   exposés pour XAI                          │
                  │   Sortie : (B, 168, 64)                     │
                  └─────────────────────┬───────────────────────┘
                                        │
                  ┌─────────────────────▼───────────────────────┐
                  │  BLOC 5 — Residual + Global Avg Pooling     │
                  │                                             │
                  │   R = LN_out + Attention_out  (skip)        │
                  │   r̄ = mean(R, dim=temps)                   │
                  │                                             │
                  │   Sortie : (B, 64)                          │
                  └──────────────┬──────────────────┬───────────┘
                                 │                  │
                  ┌──────────────▼─────────┐   ┌────▼──────────────────────┐
                  │ BLOC 6a — Classifier   │   │ BLOC 6b — Reconstruction  │
                  │ MLP : 64→32→16→1       │   │ Decoder : 64→32→11        │
                  │ + Sigmoid              │   │ (entraînement seulement)  │
                  │                        │   │                           │
                  │ ŷ ∈ [0, 1]             │   │ X̂ ∈ ℝ^(B, 168, 11)        │
                  │ Probabilité de panne   │   │ Reconstruction de X       │
                  └────────────────────────┘   └───────────────────────────┘
```

---

## 7. Pourquoi cette architecture fonctionne (résumé)

| Choix de conception | Justification | Impact mesuré |
|---------------------|--------------|---------------|
| **BiLSTM** (et non LSTM simple) | Capture le contexte avant ET après chaque instant (utile car une panne se voit dans les heures qui précèdent ET qui suivent un signal anormal) | Référence standard |
| **hidden=32, 1 couche** | Anti-overfitting : seulement 76 événements de panne dans les données réelles | AUC 0.50 → 0.62 |
| **Multi-Head Attention 4 têtes** | Chaque tête se spécialise (ex : tête 1 = pics de température, tête 2 = oscillations de pression...) | XAI interprétable |
| **Connexion résiduelle** | L'attention peut détruire le signal LSTM ; le skip préserve l'historique brut | Stabilité |
| **Global Avg Pooling** | Robuste aux séquences de longueur variable, agrège l'info sur tout l'historique | Évite le biais "dernier pas de temps" |
| **Tête de reconstruction** | Sans elle, AUC=0.50 (modèle dégénère en prédicteur de classe majoritaire) | AUC 0.50 → 0.6914 |
| **Focal Loss + pos_weight=6.9** | 7% de positifs seulement, donc pénalise plus les rares positifs ratés | Évite "prédire toujours 0" |
| **Jitter σ=0.08** | Augmentation des séquences de panne pour ↑ leur effectif | Améliore la généralisation |
| **alpha_cls=0.7** | 70% classif + 30% recon : ratio optimal trouvé par grid search | Sans recon : AUC=0.50 |

---

## 8. Récapitulatif des hyperparamètres (pour reproduire)

| Hyperparamètre | Pré-entraînement | Fine-tuning (final) |
|----------------|------------------|---------------------|
| input_dim | 11 | 11 |
| hidden_dim | 256 | **32** |
| num_layers | 3 | **1** |
| num_heads | 8 | **4** |
| bidirectional | True | True |
| dropout | 0.2 | 0.3 |
| sequence length | 168 h | 168 h |
| batch size | 64 | 64 |
| optimizer | AdamW | AdamW |
| learning rate | 1e-3 | **5e-5** |
| weight decay | 1e-5 | 1e-3 |
| scheduler | ReduceLROnPlateau (mode=min) | ReduceLROnPlateau (mode=max, sur AUC) |
| gradient clipping | max_norm = 1.0 | max_norm = 1.0 |
| loss | BCEWithLogitsLoss | 0.7·Focal(γ=2, pw=6.9) + 0.3·MSE |
| early stopping | patience=10 | patience=5 |
| epochs jusqu'à convergence | ~30 | **2** |
| total paramètres | ~5 400 000 | **~36 000** |

---

## 9. Fichiers source à consulter

- **Code modèle** : [src/models/lstm_predictor.py](src/models/lstm_predictor.py)
  - `MultiHeadAttention` (lignes 22–89)
  - `AttentionLSTM` (lignes 92–222) ← architecture principale
  - `LSTMPredictor` (lignes 225–472) ← wrapper d'entraînement
- **Wrapper multi-task** : [finetune.py:308-340](finetune.py#L308-L340) ← `MultiTaskWrapper`
- **Configuration** : [src/config/config.py](src/config/config.py) ← `ModelConfig`, `TrainingConfig`
- **Checkpoint final** : `models/lstm_predictor_finetuned.pt` (36k params, AUC=0.6914)
