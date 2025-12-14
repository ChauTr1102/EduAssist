"""
Database initialization script
Run this to create all tables in your PostgreSQL database
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import init_db, drop_db


def main():
    print("=" * 60)
    print("EduAssist Database Initialization")
    print("=" * 60)
    
    choice = input("\nWhat would you like to do?\n1. Create tables\n2. Drop all tables (DANGER!)\n3. Reset database (Drop + Create)\nChoice (1/2/3): ")
    
    if choice == "1":
        print("\nCreating database tables...")
        init_db()
        print("\n✓ Database initialized successfully!")
        
    elif choice == "2":
        confirm = input("\n⚠️  WARNING: This will delete ALL data! Type 'YES' to confirm: ")
        if confirm == "YES":
            print("\nDropping all tables...")
            drop_db()
            print("\n✓ All tables dropped!")
        else:
            print("\n✗ Operation cancelled.")
            
    elif choice == "3":
        confirm = input("\n⚠️  WARNING: This will delete ALL data! Type 'YES' to confirm: ")
        if confirm == "YES":
            print("\nDropping all tables...")
            drop_db()
            print("\nCreating database tables...")
            init_db()
            print("\n✓ Database reset successfully!")
        else:
            print("\n✗ Operation cancelled.")
    else:
        print("\n✗ Invalid choice!")


if __name__ == "__main__":
    main()
