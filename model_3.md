# Modèle 3 — DRL Optimizer (Soft Actor-Critic, SAC)

> **Rôle** : Un **agent intelligent** qui ajuste en temps réel la **fréquence VSD** et la **position de la vanne (choke)** pour maximiser la production tout en minimisant l'énergie consommée et en préservant la santé de l'équipement.
> **Algorithme** : **SAC (Soft Actor-Critic)** — RL hors-politique avec régularisation d'entropie.
> **Résultat** : Mean reward = **1039 ± 231**, soit **+73 %** vs agent aléatoire.

---

## 1. Différence fondamentale avec les Modèles 1 et 2

| Aspect | Modèle 1 | Modèle 2 | **Modèle 3** |
|--------|---------|---------|--------------|
| Paradigme | Apprentissage **supervisé** | Apprentissage **non supervisé** | **Apprentissage par renforcement** |
| Question | "Va-t-il y avoir une panne ?" | "Ce comportement est-il normal ?" | **"Quelle action est OPTIMALE maintenant ?"** |
| Données | Labels OUI/NON | Données normales | **Aucune donnée** — apprend en simulant |
| Sortie | Probabilité | Score d'anomalie | **Action continue** (freq, choke) |

> 🎯 **Différence cognitive** : les Modèles 1 et 2 **observent et diagnostiquent**. Le Modèle 3 **agit** — il prend des décisions et reçoit un feedback du monde.

---

## 2. Le cadre mathématique : MDP (Markov Decision Process)

Tout problème RL se formule comme un **MDP** : un agent qui interagit avec un environnement, étape par étape.

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

| Symbole | Signification | Pour notre ESP |
|---------|---------------|----------------|
| $\mathcal{S}$ | Espace des **états** | Les 11 capteurs + santé + cible production = 13D |
| $\mathcal{A}$ | Espace des **actions** | Δfréquence ∈ [-0.1, 0.1] et choke ∈ [0, 1] = 2D continu |
| $\mathcal{P}$ | Dynamique de transition $P(s' \mid s, a)$ | Le simulateur ESP (lois d'affinité, dégradation...) |
| $\mathcal{R}$ | Fonction de récompense $r(s, a)$ | Production + Énergie + Santé − Pénalités |
| $\gamma$ | Facteur d'actualisation | $\gamma = 0.99$ (donne du poids au futur) |

> 🎯 **Analogie** : C'est comme un **chauffeur de course** sur un circuit. À chaque virage il choisit (accélérer/freiner = action), il voit le tableau de bord (état), et il reçoit un score à la fin (récompense). L'IA apprend la meilleure politique de conduite **par l'expérience**.

---

## 3. L'espace d'état — ce que voit l'agent (13D)

À chaque pas de temps $t$, l'observation est un vecteur de 13 nombres :

$$\mathbf{s}_t = \bigl[
\underbrace{x_1, x_2, \ldots, x_{11}}_{\text{11 capteurs DHM}},\;
\underbrace{h_t}_{\text{santé}\in[0,1]},\;
\underbrace{Q^*}_{\text{cible production}}
\bigr] \in \mathbb{R}^{13}$$

| # | Variable | Source |
|---|----------|--------|
| 1–11 | motor_temp, intake_p, discharge_p, motor_current, freq, fluid_temp, voltage, vibration, current_leakage, power, diff_pressure | Capteurs simulés |
| 12 | equipment_health $h_t \in [0, 1]$ | État interne (1 = neuf, 0 = HS) |
| 13 | target_production $Q^*$ | Objectif fixé (bpd) |

**Normalisation** : Les observations sont normalisées en ligne via **VecNormalize** (moyenne + écart-type courants), ce qui stabilise SAC.

---

## 4. L'espace d'action — ce que l'agent contrôle (2D)

L'agent choisit deux variables continues à chaque pas :

$$\mathbf{a}_t = [\Delta f, \; c] \in [-0.1, 0.1] \times [0, 1]$$

| Variable | Effet sur le système |
|----------|---------------------|
| $\Delta f$ | Ajustement relatif de la fréquence VSD. **0.1 = +4 Hz** (range 40 Hz × 0.1) |
| $c$ | Position absolue du choke (0 = fermé, 1 = grand ouvert) |

> 💡 **Pourquoi continu et pas discret ?** Un choke à 0.73 produit un débit différent d'un choke à 0.74. L'agent doit pouvoir faire des **micro-ajustements** précis — c'est ce qui rend SAC particulièrement adapté (vs DQN qui est discret).

---

## 5. La fonction de récompense — le signal d'apprentissage

C'est ici que tout se joue. La récompense **encode l'objectif** : ce que l'agent doit maximiser.

$$\boxed{r_t = r_{\text{production}} + r_{\text{énergie}} + r_{\text{santé}} - r_{\text{pénalités}}}$$

### 5.1 Récompense de production (poids 1.5)

Utilise la pression différentielle comme proxy du débit :

```
prod_fraction = clip(diff_pressure / 2000, 0, 1.5)
si prod_fraction < 0.5 :  r_prod = -6.0 × (0.5 − prod_fraction)   ← pénalité forte
si 0.5 ≤ prod_fraction < 1.0 : r_prod = 1.5 × prod_fraction
si prod_fraction ≥ 1.0 :  r_prod = 1.5 × min(prod_fraction, 1.2)  ← saturé à 1.2
```

### 5.2 Récompense d'efficacité énergétique

$$r_{\text{énergie}} = 0.5 \cdot \left( \frac{\text{baseline}}{\text{power}/\text{diff\_pressure}} - 1 \right)$$

avec **baseline = 0.02 kW/psi**. Récompense positive si l'agent fait mieux que le baseline.

### 5.3 Récompense de santé (poids 0.1, réduit)

$$r_{\text{santé}} = 0.1 \cdot h_t$$

Volontairement **faible** pour que la santé n'écrase pas le signal de production.

### 5.4 Pénalités de contraintes

| Contrainte violée | Pénalité |
|-------------------|----------|
| motor_temp > 140 °C | $-5.0 \times (T - 140) / 10$ par 10 °C |
| vibration > 0.8 g | $-3.0 \times (\text{vib} - 0.8) / 0.2$ |
| current_leakage > 50 mA | $-2.0 \times (\text{leak} - 50) / 50$ |
| freq hors [20, 60] Hz | $-10.0$ |
| prod_fraction < 0.4 | $-4.0$ (anti-idle) |

> 💡 **L'art du reward shaping** : il faut **équilibrer** les signaux. Si la santé pèse trop → l'agent ralentit la pompe pour rien produire mais préserver. Si la production pèse trop → l'agent fait fondre le moteur. **Ratio production : énergie : santé ≈ 15 : 5 : 1** est la combinaison qui a fonctionné.

---

## 6. SAC — l'algorithme expliqué simplement

**SAC (Soft Actor-Critic)** combine 3 idées fortes :

### Idée 1 : Apprentissage hors-politique (off-policy)
- L'agent stocke ses expériences passées dans un **replay buffer** (100 000 transitions)
- Il **réutilise** ces expériences pour s'entraîner → très **sample-efficient**
- Avantage vs PPO (on-policy) : on n'a pas besoin de re-jouer des épisodes

### Idée 2 : Architecture Actor-Critic
- **Actor** $\pi_\phi$ : décide quelle action prendre (le décideur)
- **Critic** $Q_\theta$ : évalue la qualité d'un couple (état, action) (le juge)
- Les deux apprennent ensemble : le critic dit à l'actor s'il a bien décidé

### Idée 3 : Maximum entropy — l'exploration intelligente

Le truc unique de SAC : on n'optimise pas juste la récompense, on optimise **récompense + entropie** :

$$\boxed{J(\pi) = \sum_{t} \mathbb{E}\bigl[ r(\mathbf{s}_t, \mathbf{a}_t) + \alpha \cdot \mathcal{H}(\pi(\cdot | \mathbf{s}_t)) \bigr]}$$

où $\mathcal{H}(\pi) = -\mathbb{E}_{\mathbf{a} \sim \pi}[\log \pi(\mathbf{a}|\mathbf{s})]$ est l'**entropie de la politique**.

> 🎯 **Pourquoi ça marche ?** Plus une politique est "aléatoire" (forte entropie), plus elle **explore**. SAC trouve un équilibre automatique entre **exploiter** ce qu'il sait (haute récompense) et **explorer** de nouvelles actions (haute entropie). $\alpha$ s'ajuste tout seul pendant l'entraînement.

---

## 7. Architecture des réseaux neuronaux

SAC utilise **5 réseaux MLP** (Multi-Layer Perceptron) :

### 7.1 Actor (politique stochastique)

Une politique **gaussienne** : pour chaque état, le réseau sort la moyenne et la déviation standard d'une distribution sur les actions.

```
État (13D)
   │
   ▼
[FC 13 → 256] + ReLU
   │
   ▼
[FC 256 → 256] + ReLU
   │
   ▼
┌─────────────────────┐
│  Mean head (2D)     │ ← moyenne μ ∈ ℝ²
│  Log-std head (2D)  │ ← log écart-type σ ∈ ℝ²
└─────────────────────┘
```

À l'inférence : on tire $\mathbf{a} \sim \tanh(\mathcal{N}(\mu, \sigma^2))$ (le `tanh` borne les actions dans [-1, 1]).

### 7.2 Twin Critics (deux Q-networks)

SAC utilise **DEUX** critiques en parallèle pour **réduire le biais d'overestimation**.

```
État (13D) ∥ Action (2D) → [15D]
   │
   ▼
[FC 15 → 256] + ReLU
   │
   ▼
[FC 256 → 256] + ReLU
   │
   ▼
[FC 256 → 1] → Q(s, a) ∈ ℝ
```

On a deux copies indépendantes : **Q₁** et **Q₂**. Quand on évalue, on prend le **minimum** des deux → estimation pessimiste, donc plus stable.

### 7.3 Target Critics (copies retardées)

Pour stabiliser l'apprentissage, on garde des **copies cibles** des critiques :

$$\theta_{\text{target}} \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta_{\text{target}}, \quad \tau = 0.005$$

C'est une **mise à jour douce** ("soft update") qui empêche les Q-values d'osciller.

---

## 8. Les 3 pertes (loss functions)

### 8.1 Critic loss — apprend à évaluer

$$\mathcal{L}_Q = \mathbb{E}\bigl[\bigl(Q_\theta(\mathbf{s}, \mathbf{a}) - y\bigr)^2\bigr]$$

avec la **cible TD (Temporal Difference)** :

$$y = r + \gamma \cdot \bigl(\min_{i=1,2} Q_{\theta_i^{\text{target}}}(\mathbf{s}', \tilde{\mathbf{a}}) - \alpha \log \pi(\tilde{\mathbf{a}}|\mathbf{s}')\bigr)$$

où $\tilde{\mathbf{a}} \sim \pi(\cdot | \mathbf{s}')$ est échantillonné depuis la politique actuelle.

### 8.2 Actor loss — apprend à agir

$$\mathcal{L}_\pi = \mathbb{E}\bigl[\alpha \log \pi(\mathbf{a}|\mathbf{s}) - \min_{i=1,2} Q_{\theta_i}(\mathbf{s}, \mathbf{a})\bigr]$$

L'actor cherche à maximiser le Q-value moyen, tout en gardant de l'entropie.

### 8.3 Entropy temperature loss — auto-tune $\alpha$

$$\mathcal{L}_\alpha = -\alpha \cdot \mathbb{E}\bigl[\log \pi(\mathbf{a}|\mathbf{s}) + \mathcal{H}_{\text{target}}\bigr]$$

où $\mathcal{H}_{\text{target}} = -\dim(\mathcal{A}) = -2$ est la cible d'entropie. $\alpha$ s'ajuste pour atteindre ce niveau d'aléatoire.

---

## 9. La boucle d'apprentissage SAC (algorithme complet)

```
1.  Initialiser : Actor π_φ, Critics Q_θ1, Q_θ2, Targets Q_θ1', Q_θ2'
                  Replay buffer D (vide), entropy α (≈1.0)

2.  Phase de warmup (1000 pas) : actions aléatoires pour remplir D

3.  Boucle principale (100 000 timesteps) :
    ┌─────────────────────────────────────────────────────────────┐
    │ POUR chaque pas t :                                          │
    │                                                              │
    │   (a) Échantillonner action :                                │
    │       a_t ~ π_φ(· | s_t)                                     │
    │                                                              │
    │   (b) Exécuter dans l'environnement :                        │
    │       s_{t+1}, r_t = env.step(a_t)                           │
    │                                                              │
    │   (c) Stocker dans replay buffer :                           │
    │       D ← D ∪ {(s_t, a_t, r_t, s_{t+1})}                     │
    │                                                              │
    │   (d) Échantillonner un mini-batch (256 transitions) :       │
    │       B = sample(D, batch_size=256)                          │
    │                                                              │
    │   (e) Mettre à jour les Critics :                            │
    │       θ_1, θ_2 ← gradient descent sur L_Q                    │
    │                                                              │
    │   (f) Mettre à jour l'Actor :                                │
    │       φ ← gradient descent sur L_π                           │
    │                                                              │
    │   (g) Auto-tune α :                                          │
    │       α ← gradient descent sur L_α                           │
    │                                                              │
    │   (h) Soft update des Targets :                              │
    │       θ' ← 0.005·θ + 0.995·θ'                                │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
```

---

## 10. Schéma d'architecture (pour le dessin)

```
                        ┌────────────────────────────────────┐
                        │         AGENT (SAC)                │
                        │                                    │
                        │  ┌──────────────────────────────┐  │
                        │  │     ACTOR π_φ                │  │
                        │  │  MLP 13 → 256 → 256          │  │
                        │  │  → (mean, log_std) ∈ ℝ²×ℝ²   │  │
                        │  │  → tanh(N(μ, σ²)) → action   │  │
                        │  └─────────────┬────────────────┘  │
                        │                │ action a_t (2D)    │
                        │  ┌──────────────────────────────┐  │
                        │  │   CRITIC Q1_θ                │  │
                        │  │   MLP (13+2) → 256 → 256 → 1 │  │
                        │  └──────────────────────────────┘  │
                        │  ┌──────────────────────────────┐  │
                        │  │   CRITIC Q2_θ                │  │
                        │  │   MLP (13+2) → 256 → 256 → 1 │  │
                        │  └──────────────────────────────┘  │
                        │  ┌──────────────────────────────┐  │
                        │  │   TARGETS Q1', Q2' (soft)    │  │
                        │  └──────────────────────────────┘  │
                        │                                    │
                        │  ┌──────────────────────────────┐  │
                        │  │   REPLAY BUFFER (100k)       │  │
                        │  │   stocke (s, a, r, s')       │  │
                        │  └──────────────────────────────┘  │
                        └─────────┬──────────────────────────┘
                                  │ a_t                 ▲
                                  │              state  │ s_{t+1}
                                  ▼              reward │ r_t
                        ┌────────────────────────────────────┐
                        │   ENVIRONMENT (ESPEnvironment)      │
                        │                                    │
                        │   ┌─ Pump affinity laws            │
                        │   ┌─ Health degradation            │
                        │   ┌─ 13 sensor readings + noise    │
                        │   ┌─ Reward function:              │
                        │   │   r = prod + energy + health   │
                        │   │       - penalties              │
                        │   └─ Episode = 720 steps (30 days) │
                        └────────────────────────────────────┘
```

---

## 11. Détails de l'environnement (ESPEnvironment)

| Paramètre | Valeur |
|-----------|--------|
| Type | Gymnasium-compatible (`gym.Env`) |
| Observation space | `Box(13,)` float32 |
| Action space | `Box([-0.1, 0], [0.1, 1])` float32 |
| Steps par épisode | **720** (= 30 jours × 24 h) |
| Reset random | target_production ∈ [2000, 3000] bpd |
| Dégradation santé | $h_{t+1} = h_t - 0.0001 \cdot (1 + (f/50 - 1)^2)$ |
| Terminaison | si `equipment_health ≤ 0` (panne totale) |

---

## 12. Pourquoi VecNormalize est crucial

SAC entraîne sur des **données brutes** (températures en °C, pressions en psi, etc.). Ces échelles sont **très différentes** (température ≈ 80, pression ≈ 2000). Sans normalisation :
- Les gradients sur les capteurs à grande échelle dominent
- L'apprentissage est instable

**VecNormalize** (Stable-Baselines3) calcule en ligne la moyenne et l'écart-type de chaque dimension de l'observation, et applique :

$$\hat{s}_i = \frac{s_i - \mu_i}{\sigma_i + \epsilon}$$

Idem pour la récompense ($\gamma$-discounted return). **Effet** : convergence stable, +30 à 50 % de reward final vs sans normalisation.

> ⚠️ Note actuelle : le checkpoint `drl_optimizer.vecnorm.pkl` a été sauvegardé avec obs_dim=18 (ancienne configuration). À l'inférence, on détecte le mismatch et on **désactive la normalisation** (fallback gracieux), sans crash.

---

## 13. Hyperparamètres récapitulatifs

| Hyperparamètre | Valeur |
|----------------|--------|
| Algorithme | SAC (Soft Actor-Critic) |
| Policy | MlpPolicy (2 couches × 256) |
| Observation dim | 13 |
| Action dim | 2 |
| Learning rate | $3 \times 10^{-4}$ |
| Batch size | 256 |
| Buffer size | 100 000 |
| Learning starts | 1 000 (warmup) |
| Discount factor $\gamma$ | 0.99 |
| Soft update $\tau$ | 0.005 |
| Entropy coefficient $\alpha$ | auto-tune |
| Train frequency | 1 (à chaque pas) |
| Gradient steps | 1 (par train_freq) |
| Total timesteps | 100 000 (~139 épisodes) |
| Normalisation | VecNormalize (obs + reward) |

---

## 14. Résultats finaux

| Métrique | Valeur | Verdict |
|----------|--------|---------|
| Mean episode reward (agent) | **1 039 ± 231** | ✅ Stable |
| Mean episode reward (random) | ~600 | — |
| Gain vs random | **+73 %** | ✅ Fonctionnel |
| Convergence | ~50k timesteps | ✅ Rapide |
| Production moyenne | ~24 000 bbl / 30 jours | ⚠️ Sous-optimal (cible 75 000) |
| Énergie / baril | 0.10 kWh/bbl | ⚠️ Modèle simplifié (réel : 10–20) |

> ⚠️ **Limitations connues** :
> 1. Le modèle d'énergie de la simulation est simplifié (pas de pertes câble, pas de PVT, pas de flow multi-phase)
> 2. L'agent privilégie l'efficacité au détriment de la production
> 3. Ré-équilibrage du reward serait l'amélioration prioritaire

---

## 15. Différences clés entre les 4 modèles

| Aspect | M1 (LSTM) | M2 (AE) | **M3 (SAC)** | M4 (PINN) |
|--------|-----------|---------|--------------|-----------|
| Paradigme | Supervisé | Non supervisé | **RL** | Physics-informed |
| Données | Historique étiqueté | Données normales | **Aucune (simulation)** | Données + lois physiques |
| Type d'output | Probabilité | Score d'anomalie | **Action continue** | Variables physiques |
| Entrée temporelle | Séquence 168h | Séquence 168h | **État instantané (13D)** | Pas de temps unique |
| Architecture | BiLSTM + Attention | LSTM Encoder-Decoder | **MLP Actor + 2 Critics** | MLP profond |
| Apprentissage | Backprop sur labels | Backprop sur reconstruction | **Q-learning + policy gradient** | Backprop sur MSE + physics |

---

## 16. Fichiers source

- **Code modèle** : [src/models/drl_optimizer.py](src/models/drl_optimizer.py)
  - `ESPEnvironment` (lignes 21–312) ← simulateur Gymnasium
  - `DRLOptimizer` (lignes 315–625) ← wrapper SB3
- **Algorithme SAC** : Stable-Baselines3 (`from stable_baselines3 import SAC`)
- **Configuration** : [src/config/config.py](src/config/config.py)
- **Checkpoint** :
  - `models/drl_optimizer.zip` ← politique SAC
  - `models/drl_optimizer.vecnorm.pkl` ← stats VecNormalize

---

## 17. Phrase clé pour la soutenance

> *"Le Modèle 3 est un agent de **renforcement profond** basé sur **Soft Actor-Critic**. Contrairement aux Modèles 1 et 2 qui sont passifs (ils observent et diagnostiquent), le Modèle 3 **agit** en temps réel : à chaque pas, il choisit la fréquence VSD et la position du choke pour maximiser une récompense composite (production, efficacité, santé). Sa force vient de la **régularisation d'entropie** qui équilibre exploration et exploitation, et du **replay buffer** qui réutilise les expériences passées. Avec **+73 % de reward vs random**, l'agent démontre qu'il a appris une politique opérationnellement utile."*
