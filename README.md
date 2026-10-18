# Can Stats Reveal Your Role? — Valorant Role Classifier

Predicting a pro Valorant player's role (duelist, initiator, controller, sentinel) from their stat line,
using 2026 Valorant Champions Tour data. Built for INFO 4604 at CU Boulder.

**Write-up:** https://PranavM06.github.io/valorant-role-classifier/

**Result:** logistic regression reaches 52.5% accuracy on players it never saw during training
(4 classes; the most-common-role baseline is 29.6%). Duelists are the easiest role to identify (70%).

## Files
| File | What it is |
|---|---|
| `valorant_role_classifier.ipynb` | Full analysis: cleaning, feature engineering, clustering, model, evaluation |
| `players_stats.csv` | Source data (Kaggle: Valorant Champion Tour 2021–2026 Data) |
| `docs/` | The project web page (served by GitHub Pages) |
| `make_charts.py` | Regenerates the charts in `docs/assets/` from the notebook |

## Run it
```
pip install -r requirements.txt
jupyter notebook valorant_role_classifier.ipynb
```
