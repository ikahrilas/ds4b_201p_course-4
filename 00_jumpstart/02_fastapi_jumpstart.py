# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 0: MACHINE LEARNING & API'S JUMPSTART 
# PART 2: FAST API
# ----

# GOAL: Make stand up a basic API that makes predictions

# To Run this App:
# - Open Terminal
# - uvicorn 00_jumpstart.02_fastapi_jumpstart:app --reload
# - Navigate to localhost:8000
# - Navigate to localhost:8000/docs
# - Shutdown App: Ctrl/Cmd + C


# LIBRARIES
import pandas as pd
import polars as pl
import pycaret.classification as clf

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# SETUP ----

# Create the app object
app = FastAPI()

# Load trained Pipeline
clf_model = clf.load_model('00_jumpstart/models/xgb_model_finalized')


# 1.0 MAIN ----
@app.get("/")
async def main():
    
    content = """
    <body>
    <h1>Welcome to the email lead scoring project</h1>
    <p>Navigate to <code>docs</code> to see the API documentation</p>
    </body>
    """
    
    return HTMLResponse(content)

# 2.0 PREDICT ENDPOINT ----
@app.post("/predict")
async def predict(member_rating, country_code):
    
    # convert inputs to DataFrame
    df = pl.DataFrame(
        {
            'member_rating': [member_rating],
            'country_code': [country_code]
        }
    )
    
    # fetch predictions
    predictions_df = clf.predict_model(clf_model, data=df.to_pandas())
    
    print(predictions_df)
    
    # JSON and dictionary 
    predictions_dict = predictions_df.to_dict()
    
    return JSONResponse(content=predictions_dict)

if __name__ == '__main__':
    main()


# CONCLUSIONS ----
# * FASTAPI IS VERY EASY TO SET UP
# * BUT SOME QUESTIONS COME TO MIND...


# QUESTIONS:
# * ARE THERE OTHER API FUNCTIONS THAT WE SHOULD SET UP?
# * HOW WILL OUR USERS CONNECT WITH THIS API?
# * WHAT ABOUT SECURITY (API KEY)? 
