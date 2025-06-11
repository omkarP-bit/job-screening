# wsgi.py
from app import app

if __name__ == "__main__":
    app.run() # For local testing of this wsgi entry, Vercel won't run this directly