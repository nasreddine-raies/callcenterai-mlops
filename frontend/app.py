import streamlit as st
import requests
import os
import uuid

# Configuration de la page
st.set_page_config(page_title="Call Center AI", page_icon="🤖", layout="wide")

# URL du Router
API_URL = os.getenv("ROUTER_URL", "http://router_service:8000/predict")

# --- GESTION DE L'ÉTAT (SESSION STATE) ---

# 1. Initialiser le stockage des toutes les conversations
if "all_chats" not in st.session_state:
    # Structure: { "session_id": [ {role, content}, ... ] }
    st.session_state.all_chats = {}

# 2. Initialiser l'ID de la conversation actuelle
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.all_chats[new_id] = []

# --- SIDEBAR (MENU DE GAUCHE) ---

with st.sidebar:
    st.title("🗂️ Historique")
    
    # Bouton pour créer une nouvelle conversation
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_id
        st.session_state.all_chats[new_id] = []
        st.rerun() # Force le rechargement pour afficher la page vide

    st.divider()

    # Liste des conversations existantes
    # On affiche les clés (IDs) sous forme de liste cliquable
    chat_ids = list(st.session_state.all_chats.keys())
    
    # On inverse pour avoir les plus récents en haut (si on triait par date)
    for chat_id in reversed(chat_ids):
        # Trouver un titre (les 20 premiers caractères du premier message utilisateur)
        messages = st.session_state.all_chats[chat_id]
        if messages:
            # Chercher le premier message 'user' pour le titre
            first_user_msg = next((m['content'] for m in messages if m['role'] == 'user'), "Conversation vide")
            button_label = (first_user_msg[:25] + '...') if len(first_user_msg) > 25 else first_user_msg
        else:
            button_label = "Nouvelle conversation"

        # Si on clique sur ce bouton, on change l'ID actuel
        if st.button(button_label, key=chat_id, use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()

# --- ZONE PRINCIPALE ---

current_id = st.session_state.current_chat_id
current_messages = st.session_state.all_chats[current_id]

st.title("🤖 Call Center AI Router")

# Afficher les messages de la conversation ACTUELLE
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Décrivez votre problème..."):
    
    # 1. Ajouter le message utilisateur à l'historique actuel
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Appel API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner('Analyse et routage...'):
                response = requests.post(API_URL, json={"text": prompt})
            
            if response.status_code == 200:
                data = response.json()
                label = data.get("label", "Inconnu")
                chosen_model = data.get("chosen_model", "N/A")
                confidence = data.get("confidence", 0.0)
                
                reasoning = "Confiance élevée du Transformer." if chosen_model == "transformer" else "Modèle léger suffisant."
                
                full_response = (
                    f"**Catégorie :** `{label}`\n\n"
                    f"🛠 **Routeur :** Modèle **{chosen_model.upper()}** (Conf: {confidence:.2f})\n"
                    f"ℹ️ *{reasoning}*"
                )
                
                message_placeholder.markdown(full_response)
                
                # Ajouter la réponse à l'historique actuel
                current_messages.append({"role": "assistant", "content": full_response})
                
                # Mise à jour globale force pour que la sidebar voit le changement de titre éventuel
                st.session_state.all_chats[current_id] = current_messages
                st.rerun()
            
            else:
                message_placeholder.error(f"Erreur API: {response.status_code}")

        except Exception as e:
            message_placeholder.error(f"Erreur de connexion : {e}")