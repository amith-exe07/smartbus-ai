# SmartBus AI

Student prototype that estimates trip duration using the NYC Taxi Trip Duration dataset.

**Important limitation:** This is not a live MITS bus ETA system. It is not trained on MITS bus GPS data and does not provide live bus arrival predictions.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:10000.

## Deploy on Render
Create a Render Web Service from this GitHub repository and choose the Free plan. Render can use `render.yaml`. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`.

The displayed MAE/RMSE/R² values are from the bundle's 80/20 NYC taxi-data holdout evaluation, not from the full-data model and not from MITS buses.
