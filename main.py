from Integrations.reddit_api import RedditAPI
from Integrations.occupy_democrats import OccupyDemocratsAPI
from Integrations.gab import GabAPI
from Integrations.aljazeera import AlJazeeraAPI
from Integrations.cato_institute import CatoIntegration

from Integrations.database import Database

from NLPModel.nlpmodeller import NLPModel

import argparse

OPTIONS = ["collect", "clear", "nlp"]

# Runs all classes from integration folder
def integrations_execute(debug = True):
    # Initialize the database
    db = Database()
    
    # url, service_name, auth_method, credentials, database_file_name
    reddit_api = RedditAPI("www.reddit.com", "Reddit", "No-Auth", None, "database.db")
    reddit_api.output_basic_info()
    reddit_api.top_posts_subreddit('Libertarian')
    
    occupy_dem_api = OccupyDemocratsAPI("www.occupydemocrats.com", "OccupyDemocrats", "No-Auth", None, "database.db")
    occupy_dem_api.output_basic_info()
    left_leaning = occupy_dem_api.all_news_reports()
    
    gab_api = GabAPI("www.gab.com", "Gab", "No-Auth", None, "database.db")
    gab_api.output_basic_info()
    right_leaning = gab_api.all_posts(1874862)
    
    
    aj_api = AlJazeeraAPI("www.aljazeera.com", "Al Jazeera", "No-Auth", None, "database.db")
    aj_api.output_basic_info()
    authoritarian_leaning_opinion = aj_api.all_posts_opinion()
    authoritarian_leaning_news = aj_api.all_posts_news(400)
    
    
    cato_int = CatoIntegration("www.cato.org", "Cato Institute", "No-Auth", None, "database.db")
    cato_int.output_basic_info()
    libertarian_leaning = cato_int.all_posts()
    
    for i in libertarian_leaning:
        db.write_record("Cato Institute", i, "Libertarian", "No")

    print("Writing records...")
    
    for i in right_leaning:
        db.write_record("Gab - NationalPost", i, "Right", "No")
    
    for i in right_leaning:
        db.write_record("Gab - NationalPost", i, "Right", "No")
        
    db.clear_data()
    for i in left_leaning:
        db.write_record("OccupyDemocrats", i, "Left", "No")
    
    for i in reddit_api.top_posts_subreddit('Libertarian'):
        db.write_record("reddit", i, "Libertarian", "No")
    
    for i in authoritarian_leaning_opinion:
        db.write_record("Al Jazeera - Opinion", i, "Authoritarian", "No")
        
    for i in authoritarian_leaning_news:
        db.write_record("Al Jazeera - News", i, "Authoritarian", "No")
    
    if not debug:
        db.commit_database_changes()
    
    db.close_database_connection()

def clear_db(debug):
    if not debug:
        print("THIS FUNCTION CLEARS THE DATABASE COMPLETELY! DO NOT USE UNLESS YOU ARE CERTAIN")
        output = input("TYPE 'CONFIRM' TO START DELETION")
        if output == "CONFIRM":
            print("STARTING DELETION")
            db = Database()
            db.clear_data()
            
        else:
            print("Exiting...")
    else:
        print("Cannot clear db in debug mode...")
        
def start_nlp(debug):
    # Initialize the database
    db = Database()
    #  Initialize NLP Object
    nlp = NLPModel(db)
    nlp.start()
    nlp.validate_compulsory_fields()


# Main Function handles ArgParser and options that can be executed
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('option', choices = OPTIONS, type=str)
    parser.add_argument('--debug', type=bool, required=False)
    args = parser.parse_args()
    
    # OPTIONS = ["collect", "clear", "nlp"]
    print(args.debug)
    if args.option == "collect":
        integrations_execute(args.debug)
        
    elif args.option == "clear":
        clear_db(args.debug)
    
    elif args.option == "nlp":
        start_nlp(args.debug)

if __name__ == "__main__":
    main()