import os
import sys
from pymongo.mongo_client import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

class DatabaseConfig:
    """
    Gerencia a conexão com o MongoDB lendo as configurações
    de um arquivo .env.
    """
    def __init__(self):
        # 1. Carrega as variáveis do arquivo .env para o ambiente (os.environ)
        load_dotenv() 
        
        # 2. Lê as variáveis do ambiente, com valores padrão por segurança
        self.mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        self.db_name = os.environ.get("DB_NAME", "video_analysis_db")
        
        self.client: MongoClient = None
        self.db: Database = None
        self.video_collection: Collection = None
        
        # 3. Estabelece a conexão
        self.connect()

    def connect(self):
        """
        Estabelece a conexão com o banco de dados e inicializa as coleções.
        """
        try:
            self.client = MongoClient(self.mongo_uri)
            # Testa a conexão imediatamente
            self.client.admin.command('ping')
            print(f"Conexão com MongoDB em '{self.mongo_uri}' estabelecida com sucesso.")
        except Exception as e:
            print(f"Erro fatal ao conectar ao MongoDB em '{self.mongo_uri}': {e}", file=sys.stderr)
            raise
            
        self.db = self.client[self.db_name]
        
        # Define as coleções que este app usará
        self.video_collection = self.db["videos"]
        print(f"Conectado ao banco de dados: '{self.db_name}'")

    def get_video_collection(self) -> Collection:
        """Retorna a coleção de metadados de vídeos."""
        if self.video_collection is None:
            raise Exception("A conexão com o banco de dados não foi inicializada corretamente.")
        return self.video_collection

    def close_connection(self):
        """Fecha a conexão do cliente MongoDB."""
        if self.client:
            self.client.close()
            print("Conexão com MongoDB fechada.")