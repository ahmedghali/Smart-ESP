# Modèle 2 — LSTM Autoencoder (Détection d'anomalies)

> **Rôle** : Apprendre à quoi ressemble le **fonctionnement NORMAL** de la pompe ESP. Si une nouvelle séquence est mal reconstruite → c'est une anomalie.
> **Architecture** : Encodeur LSTM (compresse 168×11 → 16) + Décodeur LSTM (reconstruit 16 → 168×11).
> **Résultat** : **Loss reduction = −98 %** (2946 → 51.96), seuil d'alerte calibré au 95e percentile.

---

## 1. Principe fondamental — un détecteur "non supervisé"

**Différence clé avec le Modèle 1** :
- **Modèle 1 (LSTM Predictor)** : on lui donne des exemples **étiquetés** (panne / pas panne) et il apprend à classifier.
- **Modèle 2 (Autoencoder)** : on ne lui donne **QUE des exemples normaux**. Il n'a jamais vu de panne pendant l'entraînement.

**Logique** :
```
Si le modèle voit des données normales         → il les reconstruit BIEN  → erreur faible  → tout va bien
Si le modèle voit des données anormales        → il ne sait pas les copier → erreur forte   → ALERTE
```

> 🎯 **Analogie** : C'est comme un employé qui ne connaît que la routine quotidienne du bureau. Quand quelque chose d'inhabituel se passe (un dégât d'eau, un cambriolage…), il **n'arrive pas à le décrire** parce qu'il n'a jamais vu ça. Son incapacité à "rejouer la scène" → c'est l'**indicateur d'anomalie**.

---

## 2. Vue d'ensemble du flux de données

```
ENTRÉE                                                   SORTIE
┌───────────────────┐                                    ┌───────────────────┐
│ Séquence 168 h    │                                    │ X̂ — reconstruction│
│ × 11 capteurs     │ ──── ENCODEUR ─→ z ─→ DÉCODEUR ──→ │ (B, 168, 11)      │
│ (B, 168, 11)      │      compresse     reconstruit     │                   │
└───────────────────┘      vers 16D                      └───────────────────┘
        ▲                                                         │
        │                                                         ▼
        └────────────────── MSE(X, X̂) ──────────────────→ score d'anomalie
                                                          (si > seuil → ALERTE)
```

---

## 3. Architecture détaillée — bloc par bloc

### 📐 Tableau récapitulatif des formes

| # | Bloc | Entrée | Sortie | Paramètres |
|---|------|--------|--------|------------|
| 0 | Input | — | `(B, 168, 11)` | 0 |
| **ENCODEUR** | | | | |
| 1 | LSTM Encoder (2 couches, h=128) | `(B, 168, 11)` | `(B, 168, 128)` + `h_n` | ~**203 000** |
| 2 | Take last hidden state `h_n[-1]` | `(layers, B, 128)` | `(B, 128)` | 0 |
| 3 | FC Latent | `(B, 128)` | `(B, 16)` ← **bottleneck** | 128×16 + 16 = **2 064** |
| **DÉCODEUR** | | | | |
| 4 | FC Expand | `(B, 16)` | `(B, 128)` | 16×128 + 128 = **2 176** |
| 5 | Repeat sur 168 pas | `(B, 128)` | `(B, 168, 128)` | 0 |
| 6 | LSTM Decoder (2 couches, h=128) | `(B, 168, 128)` | `(B, 168, 128)` | ~**263 000** |
| 7 | FC Output | `(B, 168, 128)` | `(B, 168, 11)` | 128×11 + 11 = **1 419** |
| | **TOTAL** | | | **~471 000** |

⚠️ **Note** : ce modèle est **beaucoup plus gros** que le LSTM Predictor (471k vs 36k) parce qu'il a besoin de capacité pour reconstruire chaque pas de temps en détail, pas juste produire un scalaire.

---

## 4. Détail mathématique bloc par bloc

### 🟦 Bloc 1 — LSTM Encoder (2 couches, hidden=128)

Un LSTM **profond et monodirectionnel** qui lit la séquence et compresse progressivement l'information.

À chaque pas de temps $t$, deux couches LSTM empilées :

$$
\begin{aligned}
\mathbf{h}_t^{(1)} &= \text{LSTM}_1(\mathbf{x}_t, \mathbf{h}_{t-1}^{(1)}) \in \mathbb{R}^{128}\\
\mathbf{h}_t^{(2)} &= \text{LSTM}_2(\mathbf{h}_t^{(1)}, \mathbf{h}_{t-1}^{(2)}) \in \mathbb{R}^{128}
\end{aligned}
$$

Après 168 pas, on récupère le **dernier état caché** $\mathbf{h}_n = \mathbf{h}_{168}^{(2)} \in \mathbb{R}^{128}$.

**Pourquoi 2 couches ?** Pour capturer des dynamiques temporelles hiérarchiques — la 1ère couche apprend les patterns courts (heure par heure), la 2ème couche les patterns longs (jour par jour).

---

### 🟦 Bloc 2 — Prendre uniquement le dernier état

$$\mathbf{h}_n \in \mathbb{R}^{128}$$

C'est le **résumé complet** des 168 heures, condensé en 128 nombres. On jette les états intermédiaires car ils ne servent qu'à la propagation interne.

---

### 🟦 Bloc 3 — FC Latent : le **bottleneck**

Une couche linéaire qui **force** la compression de 128 → 16 dimensions :

$$\mathbf{z} = \mathbf{W}_z \mathbf{h}_n + \mathbf{b}_z \in \mathbb{R}^{16}$$

> 🎯 **Le rôle du bottleneck** : c'est l'**étranglement** qui force le modèle à apprendre. Si on n'avait pas ce goulot, l'autoencoder copierait juste l'entrée vers la sortie (fonction identité) sans rien apprendre.
>
> Avec seulement **16 nombres** pour représenter 168×11 = 1848 valeurs (compression **115×**), le modèle est forcé d'extraire l'**essentiel** de ce qui définit le fonctionnement normal.

---

### 🟩 Bloc 4 — FC Expand : remonter à la dimension cachée

$$\mathbf{h}_{dec} = \mathbf{W}_h \mathbf{z} + \mathbf{b}_h \in \mathbb{R}^{128}$$

Symétrique à l'encodeur. Reconvertit le résumé latent en représentation cachée.

---

### 🟩 Bloc 5 — Repeat sur 168 pas

On **duplique** le vecteur $\mathbf{h}_{dec}$ pour créer une séquence d'entrée pour le décodeur :

$$\text{Decoder input} = [\mathbf{h}_{dec}, \mathbf{h}_{dec}, \ldots, \mathbf{h}_{dec}] \in \mathbb{R}^{168 \times 128}$$

> 🎯 **Pourquoi répéter ?** Le décodeur LSTM a besoin d'une séquence d'entrée pour produire une séquence de sortie. On lui donne le **même vecteur de contexte** à chaque pas, et c'est lui qui apprend à produire **des sorties différentes** à chaque pas (en utilisant son état interne).

---

### 🟩 Bloc 6 — LSTM Decoder (2 couches, hidden=128)

Le décodeur génère pas à pas la séquence reconstruite. Initialisé avec :

$$\mathbf{h}_0^{dec} = \mathbf{h}_{dec} \;\;\text{(répété 2 fois pour les 2 couches)}, \quad \mathbf{c}_0^{dec} = \mathbf{0}$$

Puis 168 pas de LSTM standard :

$$\hat{\mathbf{h}}_t^{dec} = \text{LSTM}_{dec}(\mathbf{h}_{dec}, \hat{\mathbf{h}}_{t-1}^{dec}) \in \mathbb{R}^{128}$$

---

### 🟩 Bloc 7 — FC Output : projection vers 11 capteurs

$$\hat{\mathbf{x}}_t = \mathbf{W}_o \hat{\mathbf{h}}_t^{dec} + \mathbf{b}_o \in \mathbb{R}^{11}$$

Pour chaque pas de temps, on produit une **estimation** des 11 valeurs de capteurs.

---

## 5. La fonction de coût — MSE

$$\boxed{\mathcal{L} = \frac{1}{B \cdot 168 \cdot 11} \sum_{b, t, c} (x_{b,t,c} - \hat{x}_{b,t,c})^2}$$

C'est juste l'**erreur quadratique moyenne** entre l'entrée et la reconstruction.

**Pas de Focal Loss, pas de pos_weight** ici → on n'a pas de classification, on a juste un objectif "copie le mieux possible".

---

## 6. Détection d'anomalie — le seuil

Une fois entraîné, pour chaque nouvelle séquence $\mathbf{X}$ :

### Étape 1 : Calculer le score d'anomalie

$$\mathcal{A}(\mathbf{X}) = \frac{1}{168 \cdot 11} \sum_{t, c} (x_{t,c} - \hat{x}_{t,c})^2$$

### Étape 2 : Comparer au seuil

$$\text{Anomalie} \iff \mathcal{A}(\mathbf{X}) > \tau$$

### Étape 3 : Définition du seuil $\tau$

Le seuil est fixé au **95e percentile** des erreurs sur les données d'entraînement (normales) :

$$\tau = \text{Percentile}_{95}\bigl(\{\mathcal{A}(\mathbf{X}_i) : \mathbf{X}_i \in \text{train}\}\bigr) \approx 1.173$$

> 💡 **Interprétation** : Sur les données normales, 95 % des séquences sont reconstruites avec une erreur < $\tau$. Donc si une nouvelle séquence dépasse $\tau$, elle est plus "bizarre" que 95 % du fonctionnement normal → on alerte.

> 🎛️ **Le seuil est ajustable par l'opérateur** :
> - 99e percentile → moins de fausses alertes mais plus de pannes ratées
> - 90e percentile → plus d'alertes mais aussi plus de bruit

---

## 7. Explicabilité — quel capteur a déclenché l'alerte ?

L'autoencoder peut aussi dire **POURQUOI** il alerte, en calculant l'erreur **par capteur** :

$$\mathcal{A}_c(\mathbf{X}) = \frac{1}{168} \sum_{t=1}^{168} (x_{t,c} - \hat{x}_{t,c})^2, \quad c \in \{1, \ldots, 11\}$$

Le capteur avec la plus grosse erreur = **celui qui se comporte anormalement**.

> 🎯 **Exemple terrain** : Si le modèle alerte et que $\mathcal{A}_1$ (motor_temperature) est anormalement haute, l'opérateur sait : "L'anomalie vient du moteur, va inspecter la température".

---

## 8. Pipeline d'entraînement

```
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 — Pré-entraînement sur données synthétiques       │
│  ────────────────────────────────────────────────          │
│  • Données : UNIQUEMENT les séquences normales              │
│  • Filtre : exclure 72 h avant ET 72 h après chaque panne  │
│  • Loss : MSE                                               │
│  • Optimizer : Adam (lr=1e-3)                               │
│  • Early stopping : patience=10 sur val_loss                │
│  • Convergence : loss passe de ~3000 à ~50 (réduction −98%)│
└─────────────────────────────────────────────────────────────┘
                              ↓
                      Transfert des poids
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 — Fine-tuning sur données réelles Z1+Z2+Z3        │
│  ────────────────────────────────────────────────          │
│  • Filtrage : margin de 72h autour des pannes               │
│  • Freeze : 50 % des couches encoder (préserver l'encodeur) │
│  • Trainable : décodeur + dernières couches encoder         │
│  • Optimizer : Adam (lr=1e-4, lr réduit pour fine-tune)     │
│  • Best val loss : 0.798 ✅                                 │
│  • Seuil calibré : 95e percentile sur train → τ ≈ 1.173    │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Schéma d'architecture (pour le dessin)

```
                  ┌─────────────────────────────────────────────┐
                  │  ENTRÉE                                     │
                  │  X ∈ ℝ^(B × 168 × 11)                       │
                  │  168 h × 11 capteurs DHM                    │
                  └─────────────────────┬───────────────────────┘
                                        │
              ┌═════════════════════════▼═══════════════════════┐
              ║   ENCODEUR                                       ║
              ║ ┌──────────────────────────────────────────────┐ ║
              ║ │  Bloc 1 — LSTM Encoder (2 couches, h=128)    │ ║
              ║ │  168 pas séquentiels                          │ ║
              ║ │  Sortie : (B, 168, 128) + h_n                │ ║
              ║ └─────────────────────┬────────────────────────┘ ║
              ║                       │ on prend h_n[-1]         ║
              ║                       ▼                           ║
              ║ ┌──────────────────────────────────────────────┐ ║
              ║ │  Bloc 2 — Last Hidden State                  │ ║
              ║ │  (B, 128)                                    │ ║
              ║ └─────────────────────┬────────────────────────┘ ║
              ║                       │                           ║
              ║ ┌─────────────────────▼────────────────────────┐ ║
              ║ │  Bloc 3 — FC Latent (BOTTLENECK)             │ ║
              ║ │  Linear: 128 → 16                            │ ║
              ║ │  z ∈ ℝ¹⁶  ← compression 115×                │ ║
              ║ └─────────────────────┬────────────────────────┘ ║
              └═══════════════════════│═══════════════════════════┘
                                      │
                                      ▼  z (latent code)
              ┌═══════════════════════│═══════════════════════════┐
              ║   DÉCODEUR            │                           ║
              ║ ┌─────────────────────▼────────────────────────┐ ║
              ║ │  Bloc 4 — FC Expand                          │ ║
              ║ │  Linear: 16 → 128                            │ ║
              ║ └─────────────────────┬────────────────────────┘ ║
              ║                       │                           ║
              ║ ┌─────────────────────▼────────────────────────┐ ║
              ║ │  Bloc 5 — Repeat (168 fois)                  │ ║
              ║ │  Sortie : (B, 168, 128)                      │ ║
              ║ └─────────────────────┬────────────────────────┘ ║
              ║                       │                           ║
              ║ ┌─────────────────────▼────────────────────────┐ ║
              ║ │  Bloc 6 — LSTM Decoder (2 couches, h=128)    │ ║
              ║ │  Sortie : (B, 168, 128)                      │ ║
              ║ └─────────────────────┬────────────────────────┘ ║
              ║                       │                           ║
              ║ ┌─────────────────────▼────────────────────────┐ ║
              ║ │  Bloc 7 — FC Output                          │ ║
              ║ │  Linear: 128 → 11                            │ ║
              ║ │  Sortie : (B, 168, 11)                       │ ║
              ║ └─────────────────────┬────────────────────────┘ ║
              └═══════════════════════│═══════════════════════════┘
                                      │
                                      ▼
                  ┌─────────────────────────────────────────────┐
                  │  RECONSTRUCTION                              │
                  │  X̂ ∈ ℝ^(B × 168 × 11)                       │
                  └─────────────────────┬───────────────────────┘
                                        │
                                        ▼
                  ┌─────────────────────────────────────────────┐
                  │  SCORE D'ANOMALIE                            │
                  │  A(X) = ‖X − X̂‖²                            │
                  │                                              │
                  │  Si A(X) > τ (≈1.173) → 🚨 ANOMALIE          │
                  └─────────────────────────────────────────────┘
```

---

## 10. Pourquoi cette architecture fonctionne

| Choix de conception | Justification | Impact |
|---------------------|---------------|--------|
| **Bottleneck très étroit (16D)** | Force le modèle à n'apprendre que l'essentiel du fonctionnement normal | Sans bottleneck, le modèle ferait juste de la copie |
| **2 couches LSTM** | Capture des dynamiques hiérarchiques (court terme + long terme) | Améliore la reconstruction |
| **Monodirectionnel** | Pas besoin du futur ici, on veut juste compresser le passé | Plus rapide qu'un BiLSTM |
| **Entraîné UNIQUEMENT sur normal** | Garantit que les anomalies seront mal reconstruites | Principe fondamental de l'AE |
| **Margin de 72h autour des pannes** | Évite que les heures pré-panne ne contaminent le training set | Améliore le seuil de détection |
| **Seuil au 95e percentile** | 5 % de fausses alertes max sur normal → bon compromis | Ajustable selon le risque |

---

## 11. Différences clés entre Modèle 1 et Modèle 2

| Aspect | Modèle 1 (LSTM Predictor) | Modèle 2 (LSTM Autoencoder) |
|--------|---------------------------|------------------------------|
| **Type d'apprentissage** | Supervisé (labels OUI/NON) | Non supervisé (juste les données) |
| **Sortie** | Une probabilité $\hat{y} \in [0, 1]$ | Une reconstruction $\hat{X} \in \mathbb{R}^{168 \times 11}$ |
| **Question répondue** | "Va-t-il y avoir une panne ?" | "Ce comportement est-il normal ?" |
| **Détecte** | Pannes prévisibles (vues à l'entraînement) | N'importe quelle anomalie (même inconnue) |
| **Données d'entraînement** | Normal + panne (étiquetés) | UNIQUEMENT normal |
| **Bidirectionnel** | Oui (BiLSTM) | Non (LSTM simple) |
| **Bottleneck** | Pas explicite | Oui, à 16 dimensions |
| **Métrique d'évaluation** | AUC ROC | Loss reduction (%) + seuil sur erreur |
| **Paramètres** | ~36 000 | ~471 000 |

> 💡 **Les deux modèles sont COMPLÉMENTAIRES** :
> - Le **Modèle 1** détecte des **pannes connues** (modes de panne déjà vus pendant l'entraînement)
> - Le **Modèle 2** détecte des **comportements inhabituels** (y compris des modes de panne jamais vus, ou des combinaisons rares)

---

## 12. Récapitulatif des hyperparamètres

| Hyperparamètre | Valeur |
|----------------|--------|
| input_dim | 11 |
| hidden_dim | 128 |
| **latent_dim** | **16** (bottleneck) |
| num_layers (encoder + decoder) | 2 chaque |
| dropout | 0.2 |
| sequence length | 168 h |
| batch size | 64 |
| optimizer | Adam |
| learning rate (pré-entraînement) | 1e-3 |
| learning rate (fine-tuning) | 1e-4 |
| loss | MSELoss |
| early stopping | patience = 10 |
| threshold_percentile | 95 |
| total paramètres | ~471 000 |

---

## 13. Résultats finaux

| Métrique | Valeur | Verdict |
|----------|--------|---------|
| Loss reduction (pré-entraînement) | **−98 %** (2946 → 51.96) | ⭐ Excellent |
| Best validation loss (fine-tuning) | **0.798** | ⭐ Bon |
| Seuil d'anomalie τ (95e percentile) | **1.173** | ✅ Calibré |
| Compression latente | **115×** (1848 → 16) | ✅ Forte compression |

---

## 14. Fichiers source

- **Code modèle** : [src/models/lstm_autoencoder.py](src/models/lstm_autoencoder.py)
  - `LSTMEncoder` (lignes 22–67)
  - `LSTMDecoder` (lignes 70–135)
  - `LSTMAutoencoder` (lignes 138–224)
  - `AnomalyDetector` (lignes 227–452) ← wrapper d'entraînement
- **Fine-tuning** : [finetune.py](finetune.py) (section autoencoder ~ligne 900+)
- **Configuration** : [src/config/config.py:83-84](src/config/config.py#L83-L84)
- **Checkpoints** :
  - `models/anomaly_detector.pt` (pré-entraîné synthétique)
  - `models/anomaly_detector_finetuned.pt` (finetuned réel, loss −98 %)

---

## 15. Phrase clé pour la soutenance

> *"Le Modèle 2 est un autoencoder LSTM avec un bottleneck à 16 dimensions, entraîné uniquement sur des séquences normales. L'idée fondamentale est qu'il apprend à reconstruire parfaitement le fonctionnement normal mais échoue sur les anomalies, et c'est précisément cet **échec mesurable** qui devient notre signal de détection. Avec un seuil calibré au 95e percentile et une loss réduite de 98 %, il complète le Modèle 1 en détectant les anomalies **non labelisées** que la classification ne peut pas voir."*
