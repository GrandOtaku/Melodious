from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import requests
import csv
import re

app = Flask(__name__)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
script_dir = os.path.dirname(__file__)
csv_file_name = "genres_musicaux_descriptions.csv"
tmp_file_path = os.path.join(script_dir, csv_file_name)
csv_genres_path = os.path.join(script_dir, 'genres_musicaux_nom.csv')  # Assurez-vous que le chemin est correct
data = None
vectorstore = None

loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
print("Embeddings créés.")
vectorstore = FAISS.from_documents(data, embeddings)
print("Vectorstore créé avec succès.")

def load_genres_from_csv(csv_file_path):
    genres = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            genres.append(row[0])
    return genres

def initialize_app():
    global data, known_genres
    known_genres = load_genres_from_csv(csv_genres_path)
    print("Known genre loadé avec succès.")

def get_spotify_token(client_id, client_secret):
    url = "https://accounts.spotify.com/api/token"
    response = requests.post(url, data={"grant_type": "client_credentials"}, auth=(client_id, client_secret))
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception("Failed to retrieve token")
    
def find_spotify_playlists_by_genre(access_token, genre):
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": f"{genre}", "type": "playlist", "limit": 10}  # Cherche un peu plus de playlists pour avoir de la marge
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        playlists = response.json()['playlists']['items']
        filtered_playlists = [playlist for playlist in playlists if genre.lower() in playlist['name'].lower()]
        # Limiter les résultats à 3 playlists maximum
        limited_playlists = filtered_playlists[:3]  # Prend les trois premières playlists filtrées
        return [{"name": playlist['name'], "url": playlist['external_urls']['spotify']} for playlist in limited_playlists]
    else:
        raise Exception("Failed to search playlists")

def filter_response_for_music_theme(response):
    music_keywords = [
    "musique", "chanson", "instrument", "note", "mélodie", "concert", 
    "harmonie", "rythme", "genre musical", "clip", "BPM", "artiste",
    "orchestre", "symphonie", "opéra", "solo", "accord", "partition", 
    "compositeur", "interprète", "production musicale", "studio d'enregistrement", 
    "mixage", "mastering", "sample", "beat", "riff", "guitare", 
    "piano", "batterie", "basse", "synthétiseur", "DJ", "performance live",
    "festival de musique", "critique musicale", "échelle musicale", "tempo", 
    "clef musicale", "signature rythmique", "improvisation", "technique vocale", 
    "chorale", "bande originale", "score", "soundtrack", "licence musicale", "droits d'auteur musicaux", 
    "distribution musicale", "promotion musicale", "playlist", "streaming musical", 
    "téléchargement de musique", "vinyle", "CD", "cassette audio", "radio", 
    "mélomane", "auditeur", "fan de musique", "concert en direct", "tournée", 
    "billetterie musicale", "merchandising musical", "biographie d'artiste", "histoire de la musique",
    "théorie musicale", "acoustique", "éducation musicale", "critère de succès musical", 
    "récompenses musicales", "Grammy", "Billboard", "charts musicaux", "MTV Music Awards"
    ]

    return any(keyword in response.lower() for keyword in music_keywords)

def detect_genre_in_query(query):
    query = query.lower()
    for genre in known_genres:
        genre_pattern = '\\b' + '.?\\s*'.join(re.escape(char) for char in genre.lower()) + '.?\\b'
        if re.search(genre_pattern, query):
            print(genre)
            return genre
    return None

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    if not openai_api_key or not vectorstore:
        return jsonify({"error": "API key or vectorstore not configured."}), 400

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever())

    history = []  # Ceci devrait être stocké et géré correctement pour chaque utilisateur

    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))

    genre_detected = detect_genre_in_query(query)

    if filter_response_for_music_theme(result["answer"]) or genre_detected:
        response_text = result["answer"]
        if genre_detected:
            print(f"Genre détecté : {genre_detected}")
            try:
                access_token = get_spotify_token(spotify_client_id, spotify_client_secret)
                print(f"Token d'accès Spotify loadé")
                playlists = find_spotify_playlists_by_genre(access_token, genre_detected)
                print(f"Playlists trouvées")
                if playlists:
                    response_text += "\nVoici quelques playlists qui pourraient vous intéresser :"
                    for playlist in playlists:
                        response_text += f"\n- {playlist['name']}: {playlist['url']}"
            except Exception as e:
                print(f"Erreur lors de la recherche de playlists Spotify: {e}")
        return jsonify({"answer": response_text})
    else:
        return jsonify({"answer": "La réponse n'est pas suffisamment liée au thème musical. Veuillez essayer avec une question différente."})

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True)