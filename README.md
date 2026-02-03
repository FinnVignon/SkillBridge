# SkillBridge — Prototype

SkillBridge est un prototype de chatbot pour le matching stage/alternance basé sur les référentiels RNCP.

## Prérequis
- Python 3.10+ recommandé

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l’application
```bash
python3 app/app.py
```
Puis ouvrir :
```
http://127.0.0.1:5000
```

## Fonctionnalités
- Page d’accueil avec choix du mode (Apprenant / École / Employeur / CV)
- Chatbot guidé (questions par rôle)
- Import de CV PDF et matching RNCP
- Résultats classés par correspondances fortes/proches

## Données
Les CSV RNCP doivent être présents dans :
```
export-fiches-csv-2026-02-02/
```

## Remarques
- Le parsing CV fonctionne sur des **PDF textuels** (pas d’OCR).
- Pour les PDF scannés, il faudra ajouter un module OCR.

## Structure rapide
```
app/
  app.py
  templates/
  static/
export-fiches-csv-2026-02-02/
requirements.txt
```
