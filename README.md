# Farmland Supervision via Online Game Learning
**Dynamic Inspection Allocation under Remote Sensing Uncertainty**

> High school research project · Supervised by Prof. David Woodruff (CMU)  
> Status: In progress · Started: April 2026

---

## Research Question

China's high-standard farmland (WFF) construction has demonstrated measurable policy effectiveness (Jiang et al., 2026; Li et al., 2026), yet post-construction compliance monitoring remains a critical bottleneck. Audit results from 2025 revealed damaged irrigation facilities in 35 counties across 20 provinces — some out of service for up to four years.

**This project asks:** Under information asymmetry between regulators and contractors, how can limited inspection resources be allocated optimally to detect non-agricultural encroachment and quality fraud in farmland construction?

We propose a three-layer system:

1. **Physical sensing** — Sentinel-2 satellite imagery + NDVI analysis to quantify per-grid violation probability `p(i)` and observational uncertainty `σ(i)`
2. **AI detection** — CNN model to identify non-agricultural structures and abandoned land from satellite imagery, generating violation label sequences as proxy for compliance history
3. **MWU optimization** — Multiplicative Weights Update algorithm to dynamically allocate inspection weights based on violation history, maximising detection efficiency under budget constraints

**Theoretical grounding:** Stiglitz (1970/2001) asymmetric information → game-theoretic regulator/contractor model → MWU online learning → Hart & Holmström (1987) incentive compatibility

---

## Key Results (updated as research progresses)

| Module | Status | Output |
|--------|--------|--------|
| NDVI physical layer | In progress | Violation probability map with confidence intervals |
| CNN detection | Planned | Violation label sequences per region |
| MWU simulation | Planned | Efficiency gain vs. random inspection (%) |
| Policy analysis | Planned | ESG transparency implications |

---

## Repository Structure

```
farmland-supervision/
│
├── data/
│   ├── download_sentinel2.py      # Sentinel-2 data download via GEE API
│   ├── sample_regions.geojson     # Target farmland coordinates
│   └── README_data.md             # Data sources and licenses
│
├── notebooks/
│   ├── 01_ndvi_analysis.ipynb     # NDVI calculation + uncertainty quantification
│   ├── 02_violation_mapping.ipynb # Per-grid violation probability p(i)
│   └── 03_visualization.ipynb     # Maps and figures for report
│
├── cnn/
│   ├── train_classifier.py        # CNN binary classifier (farmland / non-agricultural)
│   ├── predict_violations.py      # Generate violation label sequences
│   ├── model_weights/             # Saved model checkpoints
│   └── README_cnn.md              # Model architecture and training details
│
├── mwu/
│   ├── game_model.py              # Regulator/contractor payoff matrix
│   ├── mwu_simulation.py          # MWU weight update algorithm
│   ├── baseline_random.py         # Random inspection baseline
│   ├── efficiency_comparison.py   # MWU vs random: detection rate analysis
│   └── README_mwu.md              # Algorithm derivation notes
│
├── report/
│   ├── draft_v1.md                # Working paper draft
│   ├── figures/                   # Auto-generated from notebooks
│   └── references.bib             # Bibliography
│
├── notes/
│   ├── theory_chain.md            # Personal notes: Stiglitz → game theory → MWU
│   ├── woodruff_lecture_log.md    # Weekly: lecture content → research action
│   └── literature_review.md      # Reading notes: Jiang2026, Li2026, Akerlof1970
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Theoretical Framework

```
Stiglitz (1970)              Jiang et al. (2026)        Li et al. (2026)
Information asymmetry   →    WFF effect decays     →    Regional gradient:
in quality markets           post-construction          East > Central > West
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    │
                          RESEARCH GAP: micro-level
                        real-time compliance monitoring
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
             Physical layer    Algorithm layer   Optimization layer
             Sentinel-2 NDVI   CNN detection     MWU inspection
             uncertainty σ(i)  violation labels  weight allocation
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                          Policy layer (ESG)
                    Hart & Holmström incentive compatibility:
                    higher inspection probability → rational
                    contractors self-select compliance
```

---

## Woodruff Lecture Log

| Lecture | Topic | Research Action | Status |
|---------|-------|-----------------|--------|
| 1–2 | Game theory + linear programming | Build payoff matrix | ✓ Done |
| 3–4 | Online learning + MWU | Implement MWU simulation | In progress |
| 5–6 | Gradient descent + backpropagation | Understand CNN training | Planned |
| 7 | ML models + GANs | Note GAN–MWU structural analogy | Planned |
| 8–10 | CNN / ViT / computer vision | Train farmland classifier | Planned |

---

## Reading List

### Must master (deep read, derivations)
- Roughgarden, T. — *MWU Algorithm Lecture Notes* (Stanford, free PDF)
- Akerlof, G. (1970) — *The Market for Lemons* (4 pages, Nobel foundation)
- Osborne, M. — *An Introduction to Game Theory*, Chapter 9 (incomplete information, free PDF)

### Cite in paper (abstract + conclusion)
- Jiang, C. et al. (2026) — *Droughts, floods, and grain yield*, J. Clean. Prod. 551, 147904
- Li, Z. et al. (2026) — *High-standard farmland construction and rations' self-sufficiency*, 农业资源与环境学报 43(1)
- Hart, O. & Holmström, B. (1987) — *The Theory of Contracts* (incentive compatibility, 2-page skim)

### Reference for methodology
- Helber, P. et al. (2019) — *EuroSAT: A Novel Dataset for Land Use Classification* (CNN training data)
- Chernozhukov, V. et al. (2018) — *Double/debiased machine learning* (DML reference from Jiang et al.)

---

## How to Run

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/farmland-supervision.git
cd farmland-supervision

# Install dependencies
pip install -r requirements.txt

# Step 1: Download satellite data (requires GEE account)
python data/download_sentinel2.py

# Step 2: Run NDVI analysis
jupyter notebook notebooks/01_ndvi_analysis.ipynb

# Step 3: Train CNN classifier
python cnn/train_classifier.py

# Step 4: Run MWU simulation
python mwu/mwu_simulation.py

# Step 5: Compare efficiency
python mwu/efficiency_comparison.py
```

All figures in `/report/figures/` are auto-generated by the notebooks and scripts above — the paper is fully reproducible.

---

## Data Sources

| Dataset | Source | License |
|---------|--------|---------|
| Sentinel-2 multispectral imagery | ESA / Google Earth Engine | Open (CC BY-SA) |
| EuroSAT land use labels | Helber et al. (2019) | MIT |
| China farmland statistics | Ministry of Agriculture and Rural Affairs | Public |

---

## Academic Context

This project is an independent research extension of coursework under Prof. David Woodruff (Carnegie Mellon University), applying the Multiplicative Weights Update algorithm — covered in lectures 3–4 of *Algorithms for Big Data* — to a real-world mechanism design problem in agricultural policy.

The research responds to the explicit policy recommendation in Li et al. (2026): *"综合运用卫星遥感等现代信息技术，强化对'非粮化''非农化'的监督"* (employ satellite remote sensing and modern information technology to strengthen monitoring of non-grain and non-agricultural encroachment).

---

## Contact

Maintained by: [Your Name]  
Institution: [Your School], [City], China  
Correspondence: [your.email@example.com]  

*Last updated: April 2026*