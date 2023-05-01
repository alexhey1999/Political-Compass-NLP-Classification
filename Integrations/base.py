import random
import sqlite3

# Many methods will be shared between different API Implementations and so this class can be used to allow them to inherit
class BaseAPI:
    def __init__(self, url, service_name, auth_method, credentials, database_file_name) -> None:
        self.url = url
        self.service_name = service_name
        self.auth_method = auth_method
        self.credentials = credentials
        self.database_file_name = database_file_name
        
    def output_basic_info(self):
        print("Service name: %s" % self.service_name)
        print("URL: %s" % self.url)
        print("Auth method: %s" % self.auth_method)
        print("Credentials: %s" % self.credentials)