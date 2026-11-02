# 📡 Phased Array Simulator

Simulateur interactif de **réseau d'antennes à balayage électronique** (ULA – *Uniform Linear Array*), développé en Python avec Streamlit, dans le cadre d'un projet PFA en Ingénierie RF & Antennes à la **SNRT, Rabat**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://VOTRE-APP.streamlit.app)

<!-- Ajoutez une capture d'écran dans assets/ puis décommentez la ligne suivante : -->
<!-- ![Aperçu](assets/screenshot.png) -->

## Fonctionnalités

- **Facteur de réseau** et loi de phase progressive `Δφ = -k·d·sin(θ₀)` pour le pointage électronique du faisceau
- **Tapering** (pondération d'amplitude) : Uniforme, Hamming, Hann, Blackman, Kaiser, Dolph-Chebyshev, Taylor
- **Métriques calculées** : HPBW (−3 dB), niveau des lobes secondaires (SLL), gain du réseau, détection des *grating lobes*
- **Réalisme matériel** : quantification de phase (déphaseurs 1 à 8 bits), erreurs aléatoires d'amplitude et de phase
- **Diagramme d'élément** `cos^q(θ)` (de l'isotrope au patch)
- **Vues** cartésienne et polaire, superposition avec le réseau uniforme
- **Études paramétriques** : influence de N, comparaison des fenêtres, élargissement du faisceau en balayage
- **Export CSV** du diagramme et des pondérations complexes

## Installation

```bash
git clone https://github.com/VOTRE-PSEUDO/phased-array-simulator.git
cd phased-array-simulator
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
python -m streamlit run app.py
```

L'application s'ouvre sur <http://localhost:8501>.

## Validation

Pour `N = 8`, `d/λ = 0,5`, `θ₀ = 0°`, fenêtre uniforme :

| Grandeur | Simulation | Théorie |
|---|---|---|
| HPBW | ≈ 12,8° | 0,886·λ/(N·d) ≈ 12,7° |
| SLL | ≈ −12,8 dB | −13,26 dB (réseau continu uniforme) |

Détails des formules : [`docs/THEORIE.md`](docs/THEORIE.md).

## Structure du dépôt

```
phased-array-simulator/
├── app.py               # Application Streamlit
├── requirements.txt     # Dépendances Python
├── docs/THEORIE.md      # Rappels théoriques
├── assets/              # Captures d'écran
├── .streamlit/config.toml
├── .gitignore
├── LICENSE
└── README.md
```

## Perspectives

- Réseau planaire 2D (UPA) avec carte de chaleur (θ, φ)
- Import du diagramme d'élément simulé sous HFSS / openEMS
- Couplage mutuel entre éléments

## Auteur

**Youssef El Hafiane** — Ingénieur en Systèmes de Télécommunications, Université Ibn Tofail.
PFE réalisé à la SNRT (Société Nationale de Radiodiffusion et de Télévision), Rabat.

## Licence

Distribué sous licence MIT — voir [`LICENSE`](LICENSE).
