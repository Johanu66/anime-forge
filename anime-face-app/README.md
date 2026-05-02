# AnimeForge Flask App

## Presentation du projet

AnimeForge est un projet de generation d'images anime qui combine une interface web moderne et des briques de modele IA pour offrir une experience simple, rapide et exploitable en pratique.  
L'objectif est de proposer une plateforme unique ou un utilisateur peut:

- decrire un personnage anime en texte,
- generer un rendu anime aleatoire,
- transformer un visage humain en style anime.

## Presentation de l'application Flask

L'application Flask est le coeur de la plateforme. Elle orchestre:

- la navigation entre les 3 fonctionnalites principales,
- la gestion des formulaires web et des uploads d'images,
- l'execution des fonctions de generation (`text`, `random`, `face-to-anime`),
- la sauvegarde des resultats dans `outputs/`,
- le telechargement direct des images produites.

Elle est structuree de maniere propre (templates, static, models, uploads, outputs), prete a l'emploi et concue pour etre facilement connectee a des modeles personnalises.
Le projet `anime-face` est autonome: il n'a pas besoin de code importe depuis d'autres dossiers du repository pour fonctionner.

Pour la fonctionnalite **Human Face to Anime**, le checkpoint actuellement utilise est:

- `anime-face/models/human-to-anime/best.pt`

Fonctionnalites principales:

- Text to Anime
- Random Anime Generator
- Human Face to Anime

## Lancer l'application

```bash
cd anime-face
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Puis ouvrir `http://127.0.0.1:5000`.

## Structure

```text
anime-face/
  app.py
  templates/
  static/
    css/
    js/
    images/
  models/
    text-to-anime/
    random-anime/
    human-to-anime/
  uploads/
  outputs/
```

## Brancher vos modeles personnalisés

Vous pouvez ajouter des hooks optionnels:

- `models/text-to-anime/inference.py` avec `generate(prompt, output_path)`
- `models/random-anime/inference.py` avec `generate(output_path)`
- `models/human-to-anime/inference.py` avec `transform(image_path, output_path)`

Si aucun hook n'est disponible, l'application utilise des fallbacks visuels robustes.

Pour `Human Face to Anime`, l'application utilise le module local `models/human-to-anime/inference.py` qui charge `models/human-to-anime/best.pt`, puis applique un fallback visuel si l'inference n'est pas disponible.
