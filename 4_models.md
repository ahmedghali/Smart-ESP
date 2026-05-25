# Les 4 modèles IA — principe de fonctionnement

> Synthèse technique des 4 modèles du framework ESP AI Digital Twin.
> Chaque modèle adresse un **objectif distinct et complémentaire**.

---

## 📊 Vue d'ensemble

| # | Modèle | Rôle | Paradigme | Métrique finale |
|---|--------|------|-----------|------------------|
| **1** | **LSTM Predictor** | Prédire les pannes 48 h à l'avance | Supervisé | AUC = **0.6914** |
| **2** | **LSTM Autoencoder** | Détecter les anomalies inconnues | Non supervisé | Loss reduction = **−98 %** |
| **3** | **DRL SAC** | Optimiser fréquence + choke en temps réel | Renforcement | +**73 %** vs random |
| **4** | **PINN** | Garantir la conformité physique | Physics-informed | R² = **0.9998** |

---

## 🟦 Modèle 1 — LSTM Predictor (BiLSTM + Multi-Head Attention)

**Principe** : lit les **168 dernières heures** des 11 capteurs et produit $\hat{y} \in [0,1]$ — "panne dans 48h ?". Paradigme supervisé classique.

**Architecture** (~36 k params, volontairement compact pour éviter l'overfit) :
```
Input (168, 11) → Linear(11→32) → BiLSTM(h=32, 1 layer) → LayerNorm
   → Multi-Head Attention (4 têtes, d_k=16) → Residual + Global Avg Pooling
   → Classifier MLP (64→32→16→1) + Sigmoid
   → Reconstruction Head (64→32→11) [training only — régularisateur]
```

**Technique clé** :
- **BiLSTM** : lit la séquence dans les 2 sens (patterns avant/après).
- **Multi-Head Attention** : poids exposés pour XAI.
- **Multi-task** : sans tête reconstruction, AUC s'effondre à 0.50.
- **Loss** : $\mathcal{L} = 0.7 \cdot \mathcal{L}_{focal}(\gamma{=}2, \text{pw}{=}6.9) + 0.3 \cdot \mathcal{L}_{MSE}$

**Pipeline 2 étapes** : pré-entraînement synthétique (hidden=256, 5.4M params) → fine-tuning intra-well sur Z1/Z2/Z3 (hidden=32, 36k params, AUC=0.6914).

---

## 🟪 Modèle 2 — LSTM Autoencoder (Détection d'anomalies)

**Principe** : apprend à **reconstruire le fonctionnement normal**. Mauvaise reconstruction = anomalie. Non supervisé — entraîné UNIQUEMENT sur séquences normales.

**Architecture** en forme de sablier (~471 k params) :
```
Input (168, 11) → LSTM Encoder (2 layers, h=128) → h_n[-1]
   → Linear(128→16) ← bottleneck (compression 115×)
   → Linear(16→128) → repeat 168× → LSTM Decoder (2 layers, h=128)
   → Linear(128→11) → Reconstruction (168, 11)
```

**Technique clé** :
- **Bottleneck à 16 dimensions** : force à n'apprendre que l'essentiel du normal.
- **Score d'anomalie** : $\mathcal{A}(X) = \frac{1}{168 \cdot 11} \sum_{t,c} (x_{t,c} - \hat{x}_{t,c})^2$
- **Seuil** : 95e percentile sur le train normal → $\tau \approx 1.173$.
- **Explicabilité** : erreur par capteur $\mathcal{A}_c(X)$ identifie quel capteur dérive (XAI gratuite).

**Complémentarité avec M1** : M1 ne reconnaît que les 8 modes appris. M2 détecte les anomalies **non labelisées** (modes inconnus) → filet de sécurité.

---

## 🟫 Modèle 3 — DRL SAC (Soft Actor-Critic)

**Principe** : un **agent IA** qui ajuste en temps réel $(\Delta f, c)$ pour maximiser une récompense composite. Apprend par interaction avec un simulateur — aucune donnée étiquetée.

**Architecture** :
```
AGENT (SAC)                          ENVIRONMENT (ESP)
 Actor π_φ (MLP 13→256→256)          ESP simulator (Gymnasium)
   → (μ, log σ) → tanh(N(μ,σ²))     ← state s_t (13D)
 Twin Critics Q₁, Q₂ (MLP (15)→1)   ← reward r_t
 Target networks (soft update)       Reward = prod + energy + health
 Replay buffer (100k transitions)      − penalties
```

**État** : 13D = 11 capteurs + `equipment_health` + `production_target`.
**Action** : 2D continu = ($\Delta f \in [-0.1, 0.1]$, $c \in [0, 1]$).

**Technique clé — SAC = 3 idées combinées** :
1. **Off-policy** + replay buffer → réutilisation des expériences (sample-efficient).
2. **Twin Critics** : 2 réseaux Q, on prend le min → évite l'overestimation.
3. **Maximum entropy** : $J(\pi) = \mathbb{E}[r + \alpha \mathcal{H}(\pi)]$ → équilibre exploration/exploitation automatique.

**Boucle 8 étapes** : env→state→actor→action→env→reward→buffer→sample→critics→actor.

**Performance** : mean reward = 1039 ± 231 (+73 % vs random).

---

## 🟨 Modèle 4 — PINN (Physics-Informed Neural Network)

**Principe** : prédit 3 grandeurs physiques ($\hat{Q}, \hat{P}_d, \hat{P}$) à partir des 11 capteurs **EN RESPECTANT les lois physiques** de la pompe.

**Architecture** (~168 k params, MLP simple — l'innovation est dans la loss) :
```
Input normalisé x̂ ∈ ℝ¹¹
   → PINN Block 1: Linear(11→200) + BatchNorm + tanh
   → PINN Block 2..5: Linear(200→200) + BatchNorm + tanh
   → Output Linear (200→3)
   → De-normalization → [Q̂, P̂_d, P̂] en unités physiques
```

**Technique clé — composite loss à 4 termes** :

| # | Terme | Rôle |
|---|-------|------|
| ① | $\mathcal{L}_{data}$ | MSE en espace normalisé |
| ② | $\mathcal{L}_{affinity}$ | $Q \propto N$, $H \propto N^2$, $P \propto N^3$ (λ₁=1.0) |
| ③ | $\mathcal{L}_{boundary}$ | bornes physiques via ReLU (λ₂=0.5) |
| ④ | $\mathcal{L}_{non\text{-}neg}$ | $\hat{y} \ge 0$ via ReLU (λ₃=1.0) |

**Insight critique** : **normaliser les sorties AUSSI** (StandardScaler). Sans cela, data_loss (~10⁴) écrase physics_loss (~10⁻¹), R² plafonne à 0.97. Avec normalisation sortie : les deux losses à ~10⁻³ → **R² = 0.9998**.

**Conformité physique** : 100 % non-négativité, déviation lois d'affinité < 4 %.

---

## 🔗 Pourquoi les 4 modèles sont complémentaires

| Question | Modèle qui répond |
|----------|-------------------|
| "Va-t-il y avoir une panne dans 48h ?" | **M1 — LSTM Predictor** |
| "Quelque chose d'anormal se passe (même inconnu) ?" | **M2 — Autoencoder** |
| "Quelle action prendre MAINTENANT pour optimiser ?" | **M3 — DRL SAC** |
| "Cette prédiction respecte-t-elle la physique ?" | **M4 — PINN** |

Aucun modèle ne pourrait remplacer un autre — ensemble, ils couvrent les 4 facettes de la gestion ESP industrielle : **prévision, détection, optimisation, validation**.

---

## 🎓 Phrase clé pour la soutenance

> *"Le framework intègre 4 modèles IA de paradigmes différents — supervisé (LSTM), non supervisé (Autoencoder), renforcement (SAC), et physics-informed (PINN) — orchestrés dans un Digital Twin. Chacun adresse une facette distincte du problème ESP : prédiction, détection, optimisation, et conformité physique. Leur complémentarité est garantie par leurs paradigmes orthogonaux : ce que l'un ne peut pas voir, un autre le détecte."*
