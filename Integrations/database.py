import sqlite3

class Database:
    def __init__(self):
        self.database_name = 'database.db'
        self.table = 'dataset'
        self.open_database_connection()
        self.con, self.cur = self.open_database_connection()
    
    def open_database_connection(self):
        try:
            con = sqlite3.connect(self.database_name)
            cur = con.cursor()
            return con, cur
        except Exception as e:
            print("There was an error connecting to database")
            print(e)
            
            
    def commit_database_changes(self):
        self.con.commit()
            
            
    def close_database_connection(self):
        try:
            self.con.close()
        except Exception as e:
            print("There was an error closing database")
            print(e)
            
            
    def write_record(self, source, statement, label, verified = 0):
        self.cur.execute(f"INSERT INTO {self.table} (Source, Statement, Label, Verified) VALUES (?,?,?,?)", (source, statement, label, verified))
        pass
    
    def clear_data(self):
        self.cur.execute(f'DROP TABLE IF EXISTS "{self.table}"')
        self.cur.execute(f'CREATE TABLE "{self.table}" ("ID" INTEGER,"Source" TEXT,"Statement" TEXT,"Label" TEXT, "Verified" TEXT, PRIMARY KEY("ID" AUTOINCREMENT))')