from flask import (
    Flask,
    request,
    jsonify,
    render_template,
)  # Importe les modules nécessaires de Flask
from langchain_openai import (
    OpenAIEmbeddings,
)  # Importe la classe pour les embeddings OpenAI de langchain
from langchain_openai import (
    ChatOpenAI,
)  # Importe la classe pour utiliser le Chat OpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
)  # Importe la chaîne de récupération conversationnelle de langchain
from langchain.document_loaders.csv_loader import (
    CSVLoader,
)  # Importe le chargeur CSV de langchain
from langchain_community.vectorstores import (
    FAISS,
)  # Importe le magasin de vecteurs FAISS de langchain
from dotenv import (
    load_dotenv,
)  # Importe la fonction pour charger les variables d'environnement
import os  # Importe le module os pour interagir avec le système d'exploitation
import requests  # Importe le module requests pour faire des requêtes HTTP
import csv  # Importe le module csv pour lire et écrire des fichiers CSV
import re  # Importe le module re pour les expressions régulières

app = Flask(__name__)  # Initialise l'application Flask
load_dotenv()  # Charge les variables d'environnement du fichier .env
openai_api_key = os.getenv(
    "OPENAI_API_KEY"
)  # Récupère la clé API d'OpenAI depuis les variables d'environnement
spotify_client_id = os.getenv(
    "SPOTIFY_CLIENT_ID"
)  # Récupère l'ID client Spotify depuis les variables d'environnement
spotify_client_secret = os.getenv(
    "SPOTIFY_CLIENT_SECRET"
)  # Récupère le secret client Spotify depuis les variables d'environnement
script_dir = os.path.dirname(__file__)  # Définit le répertoire du script actuel
csv_file_name = "genres_musicaux_descriptions.csv"  # Nom du fichier CSV contenant les descriptions des genres musicaux
tmp_file_path = os.path.join(
    script_dir, csv_file_name
)  # Construit le chemin complet vers le fichier CSV des descriptions
csv_genres_path = os.path.join(
    script_dir, "genres_musicaux_nom.csv"
)  # Construit le chemin complet vers le fichier CSV des noms de genres
data = None  # Initialisation des variables globales pour stocker les données
vectorstore = None  # Initialisation des variables globales pour stocker le vectorstore

# Création d'un chargeur CSV pour lire les données à partir du fichier spécifié, avec l'encodage UTF-8 et un délimiteur de virgule
loader = CSVLoader(
    file_path=tmp_file_path, encoding="utf-8", csv_args={"delimiter": ","}
)
data = loader.load()  # Chargement des données à partir du fichier CSV
embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)  # Initialisation des embeddings OpenAI avec la clé API fournie
print(
    "Embeddings créés."
)  # Affiche un message dans la console une fois les embeddings créés
vectorstore = FAISS.from_documents(
    data, embeddings
)  # Création d'un vectorstore FAISS à partir des documents et des embeddings
print(
    "Vectorstore créé avec succès."
)  # Affiche un message de succès une fois le vectorstore créé


# Fonction pour charger les genres musicaux à partir d'un fichier CSV
def load_genres_from_csv(csv_file_path):
    genres = []  # Initialisation d'une liste pour stocker les genres
    with open(
        csv_file_path, mode="r", encoding="utf-8"
    ) as csvfile:  # Ouverture du fichier CSV en mode lecture
        csvreader = csv.reader(
            csvfile
        )  # Création d'un objet reader pour lire le fichier CSV
        for row in csvreader:  # Pour chaque ligne du fichier CSV
            genres.append(
                row[0]
            )  # Ajoute le premier élément de chaque ligne (le nom du genre) à la liste des genres
    return genres  # Retourne la liste des genres


# Fonction pour initialiser l'application avec les genres connus
def initialize_app():
    global data, known_genres  # Déclare data et known_genres comme variables globales
    known_genres = load_genres_from_csv(
        csv_genres_path
    )  # Charge les genres connus à partir du fichier CSV spécifié
    print(
        "Known genre loadé avec succès."
    )  # Affiche un message de succès une fois les genres chargés


# Fonction pour obtenir un token Spotify en utilisant l'ID client et le secret client
def get_spotify_token(client_id, client_secret):
    url = "https://accounts.spotify.com/api/token"  # URL de l'API Spotify pour obtenir un token
    response = requests.post(
        url, data={"grant_type": "client_credentials"}, auth=(client_id, client_secret)
    )  # Envoie une requête POST pour obtenir un token
    if response.status_code == 200:  # Si la requête réussit
        return response.json()["access_token"]  # Retourne le token d'accès
    else:
        raise Exception(
            "Failed to retrieve token"
        )  # Lève une exception si la requête échoue


# Fonction pour chercher les playlists Spotify en se basant sur le genre detecté
def find_spotify_playlists_by_genre(access_token, genre):
    url = "https://api.spotify.com/v1/search"  # Définit l'URL de l'API de recherche Spotify
    headers = {
        "Authorization": f"Bearer {access_token}"
    }  # Prépare les en-têtes avec le token d'accès pour l'autorisation
    params = {
        "q": f"{genre}",
        "type": "playlist",
        "limit": 10,
    }  # Paramètres de la requête pour chercher des playlists correspondant au genre spécifié avec une limite de 10 résultats
    response = requests.get(
        url, headers=headers, params=params
    )  # Effectue la requête GET à l'API Spotify avec les paramètres et en-têtes définis
    if response.status_code == 200:  # Vérifie si la requête a réussi
        playlists = response.json()["playlists"][
            "items"
        ]  # Extrait les playlists de la réponse JSON
        filtered_playlists = [
            playlist
            for playlist in playlists
            if genre.lower() in playlist["name"].lower()
        ]  # Filtre les playlists pour inclure uniquement celles dont le nom contient le genre recherché
        limited_playlists = filtered_playlists[
            :3
        ]  # Limite le nombre de playlists renvoyées à 3 pour simplifier la réponse
        return [
            {"name": playlist["name"], "url": playlist["external_urls"]["spotify"]}
            for playlist in limited_playlists
        ]  # Retourne une liste de dictionnaires avec le nom et l'URL de chaque playlist sélectionnée
    else:
        raise Exception(
            "Failed to search playlists"
        )  # Lève une exception si la recherche échoue


# Définit une fonction pour filtrer la réponse basée sur la présence de mots-clés liés à la musique
def filter_response_for_music_theme(response):
    music_keywords = [
        "musique",
        "chanson",
        "instrument",
        "note",
        "mélodie",
        "concert",
        "harmonie",
        "rythme",
        "genre musical",
        "clip",
        "BPM",
        "artiste",
        "orchestre",
        "symphonie",
        "opéra",
        "solo",
        "accord",
        "partition",
        "compositeur",
        "interprète",
        "production musicale",
        "studio d'enregistrement",
        "mixage",
        "mastering",
        "sample",
        "beat",
        "riff",
        "guitare",
        "piano",
        "batterie",
        "basse",
        "synthétiseur",
        "DJ",
        "performance live",
        "festival de musique",
        "critique musicale",
        "échelle musicale",
        "tempo",
        "clef musicale",
        "signature rythmique",
        "improvisation",
        "technique vocale",
        "chorale",
        "bande originale",
        "score",
        "soundtrack",
        "licence musicale",
        "droits d'auteur musicaux",
        "distribution musicale",
        "promotion musicale",
        "playlist",
        "streaming musical",
        "téléchargement de musique",
        "vinyle",
        "CD",
        "cassette audio",
        "radio",
        "mélomane",
        "auditeur",
        "fan de musique",
        "concert en direct",
        "tournée",
        "billetterie musicale",
        "merchandising musical",
        "biographie d'artiste",
        "histoire de la musique",
        "théorie musicale",
        "acoustique",
        "éducation musicale",
        "critère de succès musical",
        "récompenses musicales",
        "Grammy",
        "Billboard",
        "charts musicaux",
        "MTV Music Awards",
    ]  # Liste des mots-clés associés à la musique
    return any(
        keyword in response.lower() for keyword in music_keywords
    )  # Vérifie si au moins un mot-clé est présent dans la réponse


# Définit une fonction pour détecter le genre musical dans la requête de l'utilisateur
def detect_genre_in_query(query):
    query = query.lower()  # Convertit la requête en minuscules pour la comparaison
    for (
        genre
    ) in (
        known_genres
    ):  # Parcourt la liste des genres connus pour trouver une correspondance
        genre_pattern = (
            "\\b" + ".?\\s*".join(re.escape(char) for char in genre.lower()) + ".?\\b"
        )  # Crée une expression régulière pour détecter le genre dans la requête
        if re.search(
            genre_pattern, query
        ):  # Recherche le genre dans la requête en utilisant l'expression régulière
            return genre  # Retourne le genre s'il est détecté
    return None  # Retourne None si aucun genre n'est détecté


def nettoyer_texte(texte):
    # Remplacer les séquences de deux sauts de ligne consécutifs par un seul
    texte_avec_un_seul_n = re.sub(r"\n\n", "\n", texte)
    # Remplacer les sauts de ligne isolés par des espaces
    texte_sans_n_isoles = texte_avec_un_seul_n.replace("\n", " ")
    # Supprimer les marqueurs de lien
    texte_sans_liens = re.sub(r"\[Lien\]", "", texte_sans_n_isoles)

    return texte_sans_liens

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json  # Récupère les données JSON envoyées par l'utilisateur
    query = data.get(
        "query"
    )  # Extrait la requête de l'utilisateur à partir des données
    if (
        not openai_api_key or not vectorstore
    ):  # Vérifie si la clé API OpenAI ou le vectorstore est configuré
        return (
            jsonify({"error": "API key or vectorstore not configured."}),
            400,
        )  # Retourne un message d'erreur si la configuration est manquante

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key
        ),
        retriever=vectorstore.as_retriever(),
    )  # Crée une chaîne de récupération conversationnelle avec les paramètres spécifiés pour OpenAI

    history = []  # Initialise l'historique de la conversation

    result = chain(
        {"question": query, "chat_history": history}
    )  # Utilise la chaîne pour générer une réponse basée sur la requête et l'historique
    history.append(
        (query, result["answer"])
    )  # Ajoute la requête et la réponse à l'historique

    genre_detected = detect_genre_in_query(
        query
    )  # Détecte un genre musical dans la requête de l'utilisateur

    if (
        filter_response_for_music_theme(result["answer"]) or genre_detected
    ):  # Filtre la réponse pour s'assurer qu'elle est liée au thème musical
        response_text = result["answer"]  # Utilise la réponse générée
        if genre_detected:  # Si un genre musical est détecté
            try:
                access_token = get_spotify_token(
                    spotify_client_id, spotify_client_secret
                )  # Obtient un token d'accès Spotify
                playlists = find_spotify_playlists_by_genre(
                    access_token, genre_detected
                )  # Trouve des playlists Spotify correspondant au genre détecté
                if playlists:  # Ajoute les playlists trouvées à la réponse
                    response_text += (
                        ". Voici quelques playlists qui pourraient vous intéresser :"
                    )
                    for playlist in playlists:
                        response_text += f" {playlist['name']}: {playlist['url']}"
            except (
                Exception
            ) as e:  # Gère les exceptions lors de la recherche de playlists Spotify
                print(f"Erreur lors de la recherche de playlists Spotify: {e}")
        response_text = nettoyer_texte(response_text)
        return jsonify(
            {"answer": response_text}
        )  # Retourne la réponse enrichie avec les playlists Spotify
    else:
        return jsonify(
            {
                "answer": "La réponse n'est pas suffisamment liée au thème musical. Veuillez essayer avec une question différente."
            }
        )  # Retourne une réponse indiquant l'absence de lien avec le thème musical


# Point d'entrée principal pour exécuter l'application Flask en mode debug
if __name__ == "__main__":
    initialize_app()  # Appelle la fonction d'initialisation de l'application
    app.run(debug=True)  # Lance le serveur Flask en mode debug
