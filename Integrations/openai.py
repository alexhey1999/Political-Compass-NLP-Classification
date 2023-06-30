import sqlite3
import openai
import os
from dotenv import load_dotenv
from alive_progress import alive_bar
import random
import time

class OpenAIConverter:
    def __init__(self, raw_database):
        self.database_name = 'databaseopenai.db'
        self.original_database = raw_database
        self.table = 'datasetopenai'
        self.open_database_connection()
        self.con, self.cur = self.open_database_connection()
        self.random_seed = 10000
        random.seed(self.random_seed)
        self.x_split_cats = ["Left", "Right"]
        self.y_split_cats = ["Authoritarian", "Libertarian"]
        
        
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
    
    def clear_data(self):
        self.cur.execute(f'DROP TABLE IF EXISTS "{self.table}"')
        self.cur.execute(f'CREATE TABLE "{self.table}" ("ID" INTEGER,"Source" TEXT,"Statement" TEXT, "OriginalStatement" TEXT, "Label" TEXT, "Verified" TEXT, PRIMARY KEY("ID" AUTOINCREMENT))')
    
            
    def write_record(self, source, statement, original_statement, label, verified = 0):
        self.cur.execute(f"INSERT INTO {self.table} (Source, Statement, OriginalStatement, Label, Verified) VALUES (?,?,?,?,?)", (source, statement, original_statement, label, verified))
        pass
    
        
    def get_all_data(self):
        res = self.cur.execute(f"SELECT * FROM {self.table}")
        data = res.fetchall()
        return data
    
    def load_data(self):
        data = self.original_database.get_all_data()
        self.classes = {}
        for row in data:
            if row[3] not in self.classes:
                self.classes[row[3]] = 1
            else:
                self.classes[row[3]] += 1
        random.shuffle(data)
        balanced_data = self.balance_data(data, self.classes)
        return balanced_data
             
   
    def balance_data(self, data, item_dict):
        running_record_counts = {}
        # Balance the dataset by taking the category with the minimum data and then reducing the data to the correct size.
        max_record_value = min(item_dict.values())
        reduced_data = []
        for i in data:
            if i[3] not in running_record_counts:
                running_record_counts[i[3]] = 1
            else:
                running_record_counts[i[3]] += 1
                
            if running_record_counts[i[3]] < max_record_value:
                reduced_data.append(i)
        
        libertarian = [a for a in reduced_data if a[3] == 'Libertarian']
        authoritarian = [a for a in reduced_data if a[3] == 'Authoritarian']
        left = [a for a in reduced_data if a[3] == 'Left']
        right = [a for a in reduced_data if a[3] == 'Right']
        
        final_reduced_data = []
        for li, au, le, ri in zip(libertarian, authoritarian, left, right):
            final_reduced_data.append(li)
            final_reduced_data.append(au)
            final_reduced_data.append(le)
            final_reduced_data.append(ri)
        return final_reduced_data
    
    def get_openai_records(self):
        res = self.cur.execute(f"SELECT * FROM {self.table}")
        data = res.fetchall()
        return data
    
    def run_process(self):
        load_dotenv()
        all_raw_records = self.load_data()
        all_raw_quotes = [a[3] for a in self.get_openai_records()]
        # print(all_raw_records)
        filtered_list = []
        for i in all_raw_records:
            if i[2] not in all_raw_quotes:
                filtered_list.append(i)
        all_raw_records = filtered_list
        messages_to_load = [
            "The statement following the next occurance of the delimeter '---' is news headline that ends at the following '---' delimiter. The final word of the prompt will correspond to a political category. Using these 2 pieces of information you should convert the news headline into a tweet as if it was written by an average twitter user.", 
            "The tweet must be in support of the category ideology. Libertarians uphold the belief of Liberty and personal freedoms. Authoritarians use central power to preserve the political status quo (often done through enforcement or military action).",
            "The following people are prominent political figueres followed by their standing. Use this context to base the tweets: Joe Biden - Left, Donald Trump - Right, Barack Obama - Left, George W. Bush - Right, Bill Clinton - Left, Hillary Clinton - Left, Mike Pence - Right, Ron Desantis - Right, Gavin Newsom - Left",
            "You should avoid using tags in the tweet and must not express the category in plain text, only imply the users beliefs"
            ] 
        # prompt = "The statement following the next occurance of the delimeter '---' is news headline that ends at the following '---' delimiter. The final word of this prompt will correspond to a political category. Using these 2 pieces of information you should convert the news headline into a tweet as if it was written by an average twitter user. If you understand, simply reply 'OK'"
        # #  If you do not think there is enough context to write a tweet based on the statement and category, you should populate the tweet with information within reason so that it fits the category given. Never outright declare the category of the original text in the tweet.---
        
        openai.api_key = os.getenv('OPENAI_KEY')
        all_messages = []
        for i, (_, _, original_statement, leaning, _) in enumerate(all_raw_records):
            record_context = []
            for i in messages_to_load:
                record_context.append({"role": "assistant", "content": i})
            record_context.append({"role": "assistant", "content": "---"+original_statement+"---"+leaning})
            all_messages.append(record_context)

        # for i in all_messages:
        #     print(i)
        #     input()
            
        # exit()    
        with alive_bar(len(all_messages)) as bar:
            for open_ai_message_prompt, original_record in zip(all_messages, all_raw_records):
                (_, source, original_statement, leaning, verified) = original_record
                try:
                    open_ai_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=open_ai_message_prompt)
                    self.write_record(source, open_ai_response['choices'][0]["message"]["content"], original_statement, leaning, verified)
                    self.commit_database_changes()
                    bar()
                except:
                    print("Error Connecting to OpenAI. Pausing Process...")
                    time.sleep(30)
                    bar()
                    continue
                