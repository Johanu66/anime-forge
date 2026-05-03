# AnimeForge

AnimeForge est un projet du cours **8INF887 - Apprentissage profond**.  

## 🎯 **Objectif du projet**  
L'objectif est de proposer une plateforme web unique pour la generation d'images anime via trois fonctionnalites:

1. `Text to Anime` : generation a partir d'un prompt texte.
2. `Random Anime` : generation aleatoire d'un personnage anime.
3. `Human Face to Anime` : transformation d'un visage reel en style anime.

L'interface finale est une application Flask moderne (dossier `anime-face-app`) qui integre ces trois parcours dans une experience unique.

## 👥 Equipe du projet

Projet realise par:

- Johanu GANDONOU
- Amal Ouedraogo
- Imane BOUGHELEM

## 🚀 Tester rapidement

### 🌐 Version en ligne
Application deployee: **https://anime-forge.randever.com**

### 🎬 Video de demonstration
Une demo du projet est disponible ici: [Video de presentation (Google Drive)](https://drive.google.com/file/d/1mOKIjd4gmxSNtEh59dJBb4Th89cLYWNM/view?usp=sharing)

### 📄 Rapport final
Le rapport détaillé du projet est disponible ici: [rapport-final.pdf](./rapport-final.pdf)


## 💻 Lancer en local

### 1) Aller dans l'application Flask

```bash
cd anime-face-app
```

### 2) Creer et activer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

Sous Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Installer les dependances

```bash
pip install -r requirements.txt
```

### 4) Lancer le serveur Flask

```bash
python app.py
```

Puis ouvrir `http://127.0.0.1:5000`.

## ✅ Verification fonctionnelle

Une fois l'application ouverte:

1. Ouvrir `Text to Anime`, saisir un prompt, puis cliquer sur **Generate**.
2. Ouvrir `Random Anime`, cliquer sur **Generate Random**.
3. Ouvrir `Human Face to Anime`, uploader une image visage puis cliquer sur **Transform**.

## 🧠 Modeles locaux utilises dans l'app

L'app est prete a utiliser les poids presents dans `anime-face-app/models/`:

- `models/text-to-anime/lora_weights_local/adapter_model.safetensors`
- `models/random-anime/checkpoint_epoch_80.pth.zip`
- `models/human-to-anime/best.pt`

Pour la partie `Text to Anime`, les poids LoRA sont locaux.  
Le modele de base Stable Diffusion peut etre charge localement si vous placez un modele complet dans `models/text-to-anime/base_model_local/` (avec `model_index.json`) ou via `ANIME_FACE_TEXT_BASE_MODEL`.

⚠️ Si vous voulez forcer un mode strictement local (sans telechargement), lancez:

```bash
export ANIME_FACE_TEXT_LOCAL_ONLY=1
python app.py
```

## 📚 Ressources du projet

- Application Flask: `anime-face-app/`
- Fonctionnalite generation aleatoire (code de recherche et entrainement): `random-anime/`
- Fonctionnalite texte-vers-image (code de recherche et entrainement): `text-to-anime/`
- Fonctionnalite image-vers-image (code de recherche et entrainement): `human-to-anime/`
