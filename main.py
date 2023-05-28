from Integrations.reddit_api import RedditAPI
from Integrations.occupy_democrats import OccupyDemocratsAPI
from Integrations.database import Database

def main():
    # Initialize the database
    db = Database()
    
    # # url, service_name, auth_method, credentials, database_file_name
    # reddit_api = RedditAPI("www.reddit.com", "Reddit", "No-Auth", None, "database.db")
    # reddit_api.output_basic_info()
    # reddit_api.top_posts_subreddit('Libertarian')
    
    occupy_dem_api = OccupyDemocratsAPI("www.occupydemocrats.com", "OccupyDemocrats", "No-Auth", None, "database.db")
    occupy_dem_api.output_basic_info()
    left_leaning = occupy_dem_api.all_news_reports()
    
    
    # db.clear_data()
    for i in left_leaning:
        # source, statement, label, verified = 0
        db.write_record("OccupyDemocrats", i, "Left", "No")
    
    # for i in reddit_api.top_posts_subreddit('Libertarian'):
    #     # source, statement, label, verified = 0
    #     db.write_record("reddit", i, "Libertarian", "No")
    
    db.commit_database_changes()
    db.close_database_connection()
    
if __name__ == "__main__":
    print("Start Data Collection")
    main()