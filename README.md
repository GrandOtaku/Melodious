
# IA de recommandation musicale
Ce projet est une API construite avec Flask et intégrant LanguageChain et OpenAI, spécialement conçue pour offrir des fonctionnalités autour de la musique. Elle permet de traiter des requêtes textuelles et de fournir des réponses en lien avec la musique grâce à l'intelligence artificielle.

## Configuration Initiale
Avant de lancer l'API, quelques étapes de configuration sont nécessaires.

### Prérequis
- Python 3.8 ou plus récent.
- pip (gestionnaire de paquets Python).
- Un environnement virtuel Python (recommandé).

## Installation

Cloner le repo Github :
```bash
git clone <url_du_dépôt>
cd <nom_du_dossier_du_projet>
```
Créez et activez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate
```
Installez les dépendance :
```bash
pip install -r requirements.txt
```
    
## Environment Variables

Ouvrez le fichier .env et modifiez les valeurs des variables d'environnement pour correspondre à votre configuration

`OPENAI_API_KEY` = Votre clé API OpenAI.

`SPOTIFY_CLIENT_ID` = Votre ID d'application Spotify

`SPOTIFY_CLIENT_SECRET` = Votre Client secret d'application Spotify

## Lancement
Pour démarrer l'API, utilisez la commande suivante dans le répertoire racine du projet :
```bash
  python3 API.py
```

Ceci lancera l'API Flask sur localhost avec le port par défaut 5000. Vous pouvez accéder à l'API via http://localhost:5000.
