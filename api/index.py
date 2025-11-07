from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from textblob import TextBlob
import pandas as pd
import jwt
from datetime import datetime, timedelta
import io
import os
from typing import Optional

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock user database
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "admin",  # In production, use proper password hashing
        "disabled": False,
    }
}

# Token creation function
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Authentication function
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = users_db.get(username)
    if user is None:
        raise credentials_exception
    return user

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running"}

# Login endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or user["hashed_password"] != form_data.password:
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password"
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Analyze text endpoint
@app.post("/analyze")
async def analyze_text(
    text: str = None,
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    if not text and not file:
        raise HTTPException(status_code=400, detail="No text or file provided")
    
    if file:
        if file.content_type != "text/csv":
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        contents = await file.read()
        try:
            # Read CSV file
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            
            # Assuming the text column is named 'text'
            if 'text' not in df.columns:
                return {"error": "CSV must contain a 'text' column"}
            
            # Analyze sentiment for each text
            results = []
            for _, row in df.iterrows():
                blob = TextBlob(str(row['text']))
                sentiment = 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
                results.append({
                    'text': row['text'],
                    'sentiment': sentiment,
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
            
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # If text is provided
    blob = TextBlob(text)
    sentiment = 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
    
    return {
        'text': text,
        'sentiment': sentiment,
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

# This is needed for Vercel
def handler(req: Request):
    return app

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
