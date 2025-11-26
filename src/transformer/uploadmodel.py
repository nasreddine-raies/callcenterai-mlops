from huggingface_hub import create_repo, upload_folder
from transformers import AutoModel, AutoTokenizer

# -------------------------------
# 1️⃣ Crée le repo sur Hugging Face
# Remplace "mon_modele" par le nom que tu veux donner à ton modèle
repo_url = create_repo("mon_modele", exist_ok=True)

# -------------------------------
# 2️⃣ Upload du modèle depuis ton dossier local
# Remplace "./models/transformer" par le chemin vers ton modèle sur ton PC
upload_folder(
    folder_path="./models/transformer",
    repo_id="nsayer/mon_modele",  # Remplace par ton username et le nom du repo
    repo_type="model"
)

print("Upload terminé ! Ton modèle est disponible sur :", repo_url)

# -------------------------------
# 3️⃣ Chargement du modèle depuis Hugging Face
model = AutoModel.from_pretrained("nsayer/mon_modele")
tokenizer = AutoTokenizer.from_pretrained("nsayer/mon_modele")

print("Modèle chargé avec succès !")
