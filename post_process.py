import psycopg2
import pandas as pd
import base64
import os

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Paramètres de connexion
DB_HOST = "0.0.0.0"
DB_NAME = "crawl"
DB_USER = "cookiestudy23"
DB_PASSWORD = "cookiestudy2023"
DB_PORT = "5432"

EXPERIMENT_ID = "test_2025-04-15_08-06"

# Requête SQL
SQL_QUERY = """
select cookies_with_predictions.name, cookies_with_predictions.value, cookie_domain, collection_strategy, websites.name as webname, websites.screenshot \
from cookies_with_predictions \
inner join websites on cookies_with_predictions.visit_id = websites.visit_id \
where  cookies_with_predictions.collection_strategy!='Accept' and cookies_with_predictions.collection_strategy!='Save cookie settings' and websites.experiment_id = '"""+  EXPERIMENT_ID +"""' and position( websites.name in cookie_domain  )=0 and position( cookie_domain in websites.name )=0 and position( 'test' in lower(cookies_with_predictions.name) )=0 and position( '_cf' in cookies_with_predictions.name )=0 and position( 'consent' in lower(cookies_with_predictions.name) )=0 and length(cookies_with_predictions.value)>5 ;;
"""

# Charger le fichier CSV (supposons qu'il s'appelle paragraphs.csv)
csv_df = pd.read_csv("paragraphs.csv",sep=';')  # Doit avoir les colonnes correspondantes à id2

# Connexion à la base de données
conn = psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT
)

cur = conn.cursor()
cur.execute(SQL_QUERY)
rows = cur.fetchall()
#print(str(rows))
colnames = [desc[0] for desc in cur.description]


html_output = """<html><head><meta charset='utf-8'></head><body>"""


for row in rows:
    row_dict = dict(zip(colnames, row))

    website = row_dict["webname"]  # première colonne (ex: id1)
    strategy = row_dict["collection_strategy"]  # deuxième colonne (utilisée pour choisir la colonne dans le CSV)
    screenshot = row_dict["screenshot"]  # dernière colonne supposée être l'image (BYTEA PNG)

    # Encoder l'image PNG en base64 pour affichage HTML
    if screenshot:
        picture_b64 = base64.b64encode(screenshot).decode('utf-8')
        picture_tag = f"<img src='data:image/png;base64,{picture_b64}' alt='Image'/><br>"
    else:
        picture_tag = "<p>[Image non disponible]</p>"

    # Cherche le paragraphe en utilisant id2 comme nom de colonne
    print(strategy)
    if str(strategy) in csv_df.columns:
        paragraph = csv_df[str(strategy)].iloc[0]  # ou une autre logique d'extraction
    else:
        paragraph = "[Paragraphe non trouvé]"
    print(paragraph)
    # Format du message HTML
    message = f"""
    <div style='margin-bottom: 40px;'>
        <p>Bonjour {website},</p>
        <p>Comment allez-vous?</p>
        <p>{paragraph}</p>
        {picture_tag}
    </div>
    """
    #print(message)

    html_output = message
    html_output += "</bodsy></html>"
    constat_path= os.path.join("Constat_générés",f"constat_{website}.html")
    with open(constat_path, "w", encoding="utf-8") as f:
        f.write(html_output)

# Enregistrer dans un fichier HTML


cur.close()
conn.close()