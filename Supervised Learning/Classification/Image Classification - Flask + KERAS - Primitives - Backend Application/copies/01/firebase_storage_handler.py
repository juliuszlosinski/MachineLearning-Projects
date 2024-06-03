import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

"""
FirebaseHandler - Handling connection to the Firebase.
Input:
- path to configuration file,
"""
class FirebaseHandler:
    def __init__(self, path_to_config) -> None:
        """
        Initializing params.
        """
        self.config = path_to_config
        self.firebase_credentials = credentials.Certificate(path_to_config)
    def connect(self):
        """
        Connecting to the firestore.
        """
        firebase_admin.initialize_app(self.firebase_credentials)
        self.firebase_database = firestore.client()
        if self.firebase_database is not None:
            self.connected = True
            print(f"LOG::FirebaseHandler::Connecting to the Firebase Firestore success!")
        else:
            self.connected = False
            print(f"LOG::FirebaseHandler::Connecting to the Firebase Firestore failed!")
        return self.connected
            
    
    def post_document(self, collection_name, document_data, document_id=""):
        """
        Posting document on the specific collection.
        """
        if self.connected:
            if document_id != "":
                self.firebase_database.collection(collection_name).document(document_id).set(document_data)
            else:
                self.firebase_database.collection(collection_name).document().set(document_data)
        else:
            print(f"LOG::FirebaseHandler::Not connected!")