import os, math, joblib
import pandas as pd
from flask import Flask, request, render_template_string, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(HERE, 'smartbus_validated_model.joblib'))
model = bundle['model']
FEATURES = bundle['features']
METRICS = bundle.get('evaluation_reference', {})
app = Flask(__name__)

HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SmartBus AI</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>
*{box-sizing:border-box}body{margin:0;background:#f2f6fc;color:#14233c;font-family:Arial, sans-serif}header{background:#0c1f3d;color:#fff;padding:24px max(20px,calc((100vw - 1120px)/2))}header h1{margin:0 0 6px}header p{margin:0;color:#d1dced}main{max-width:1120px;margin:24px auto;padding:0 16px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}.metric,.panel{background:#fff;border-radius:14px;padding:18px;box-shadow:0 3px 15px #102b4b0b}.metric small{display:block;color:#61738b;margin-bottom:8px}.metric strong{font-size:22px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}h2{font-size:20px;margin:0 0 14px}#map{height:300px;border-radius:10px}.muted,.mapnote{font-size:12px;color:#61738b}.mapnote{min-height:22px;margin-top:8px}form{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{font-size:13px;font-weight:bold;display:block;margin-bottom:6px}input,select{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:8px;background:white;font:inherit}.full{grid-column:1/-1}button{width:100%;background:#2563eb;color:white;border:0;border-radius:9px;padding:13px;font-size:15px;font-weight:bold;cursor:pointer}.result{margin-top:15px;padding:18px;border-radius:12px;background:#eef5ff}.result small{font-weight:bold;color:#2563eb}.result h3{font-size:29px;margin:9px 0}.result p{margin:8px 0}.notice{background:#fff5e8;border-radius:12px;padding:16px;margin-top:18px;font-size:14px;line-height:1.5}footer{max-width:1120px;margin:auto;padding:0 16px 25px;color:#64748b;font-size:12px}@media(max-width:720px){.metrics{grid-template-columns:repeat(2,1fr)}.cols{grid-template-columns:1fr}}
</style></head><body><header><h1>🚌 SmartBus AI</h1><p>Trip-duration estimation · NYC taxi-data prototype</p></header><main>
<div class="metrics"><div class="metric"><small>Holdout MAE</small><strong>{{mae}}</strong></div><div class="metric"><small>Holdout RMSE</small><strong>{{rmse}}</strong></div><div class="metric"><small>Holdout R²</small><strong>{{r2}}</strong></div><div class="metric"><small>Model inputs</small><strong>17 features</strong></div></div>
<div class="cols"><section class="panel"><h2>Choose a route</h2><div id="map"></div><div class="mapnote" id="mapnote">Click map once for pickup, then again for drop-off.</div><p class="muted">Map centred on New York City to match the prototype dataset.</p></section>
<section class="panel"><h2>Trip details</h2><form id="form">
<div><label>Pickup latitude</label><input name="pickup_lat" type="number" step="any" value="40.75958" required></div><div><label>Pickup longitude</label><input name="pickup_lon" type="number" step="any" value="-74.00296" required></div>
<div><label>Drop-off latitude</label><input name="dropoff_lat" type="number" step="any" value="40.77112" required></div><div><label>Drop-off longitude</label><input name="dropoff_lon" type="number" step="any" value="-73.99361" required></div>
<div><label>Passengers</label><select name="passenger_count">{% for n in range(1,7) %}<option value="{{n}}">{{n}}</option>{% endfor %}</select></div><div><label>Vendor</label><select name="vendor_id"><option value="1">Vendor 1</option><option value="2">Vendor 2</option></select></div>
<div><label>Store-and-forward flag</label><select name="store_and_fwd_flag"><option>N</option><option>Y</option></select></div><div><label>Pickup hour (0–23)</label><input name="hour" type="number" min="0" max="23" value="17" required></div>
<div><label>Day of week</label><select name="dayofweek">{% for n in range(7) %}<option value="{{n}}" {% if n==2 %}selected{% endif %}>{{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][n]}}</option>{% endfor %}</select></div><div><label>Month (1–12)</label><input name="month" type="number" min="1" max="12" value="6" required></div>
<div class="full"><button type="submit">Predict Trip Duration</button></div></form><div class="result" id="result"><small>ESTIMATED TRIP DURATION</small><h3>Ready to predict</h3><p>Enter route details and click Predict.</p></div></section></div>
<div class="notice"><b>Prototype limitation:</b> This model uses NYC Taxi Trip Duration data. It is not trained on MITS bus GPS data and does not provide live bus arrivals. Estimates are not guaranteed road travel times.</div></main><footer>SmartBus AI student prototype · Metrics are from an 80/20 NYC taxi-data holdout evaluation, not live bus performance.</footer>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const map=L.map('map').setView([40.75,-73.99],12);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap'}).addTo(map);let pm=null,dm=null,line=null,clicks=0;
map.on('click',e=>{const f=document.querySelector('#form');if(clicks%2===0){if(pm)map.removeLayer(pm);pm=L.marker(e.latlng).addTo(map).bindPopup('Pickup').openPopup();f.elements.pickup_lat.value=e.latlng.lat.toFixed(5);f.elements.pickup_lon.value=e.latlng.lng.toFixed(5);document.getElementById('mapnote').textContent='Pickup selected. Click again for drop-off.';}else{if(dm)map.removeLayer(dm);if(line)map.removeLayer(line);dm=L.marker(e.latlng).addTo(map).bindPopup('Drop-off').openPopup();f.elements.dropoff_lat.value=e.latlng.lat.toFixed(5);f.elements.dropoff_lon.value=e.latlng.lng.toFixed(5);if(pm)line=L.polyline([pm.getLatLng(),e.latlng],{color:'#2563eb'}).addTo(map);document.getElementById('mapnote').textContent='Route selected. Adjust details and predict.';}clicks++;});
document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();const out=document.getElementById('result');out.innerHTML='<p>Calculating…</p>';const payload=Object.fromEntries(new FormData(e.target).entries());try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||'Prediction failed');out.innerHTML=`<small>ESTIMATED TRIP DURATION</small><h3>${d.minutes.toFixed(2)} minutes</h3><p>Approximately ${Math.round(d.seconds)} seconds</p><hr><p>Straight-line distance: ${d.distance_km.toFixed(3)} km</p><p>Pickup: ${d.pickup_lat.toFixed(5)}, ${d.pickup_lon.toFixed(5)}</p><p>Drop-off: ${d.dropoff_lat.toFixed(5)}, ${d.dropoff_lon.toFixed(5)}</p><p class="muted">Estimated from NYC taxi data; not a guaranteed road travel time.</p>`;}catch(err){out.innerHTML='<p><b>Could not predict:</b> '+err.message+'</p>';}});
</script></body></html>'''

def build_features(p):
    plat, plon = float(p['pickup_lat']), float(p['pickup_lon'])
    dlat, dlon = float(p['dropoff_lat']), float(p['dropoff_lon'])
    passengers, vendor = int(p.get('passenger_count',1)), int(p.get('vendor_id',1))
    flag, hour, dow, month = str(p.get('store_and_fwd_flag','N')).upper(), int(p.get('hour',17)), int(p.get('dayofweek',2)), int(p.get('month',6))
    for name,val,lo,hi in [('pickup_lat',plat,-90,90),('dropoff_lat',dlat,-90,90),('pickup_lon',plon,-180,180),('dropoff_lon',dlon,-180,180)]:
        if not lo <= val <= hi: raise ValueError(f'{name} must be between {lo} and {hi}.')
    if not 1 <= passengers <= 6: raise ValueError('Passenger count must be from 1 to 6.')
    if vendor not in (1,2): raise ValueError('Vendor must be 1 or 2.')
    if flag not in ('N','Y'): raise ValueError('Store-and-forward flag must be N or Y.')
    if not 0 <= hour <= 23: raise ValueError('Hour must be from 0 to 23.')
    if not 0 <= dow <= 6: raise ValueError('Day of week must be from 0 to 6.')
    if not 1 <= month <= 12: raise ValueError('Month must be from 1 to 12.')
    lat1,lat2=math.radians(plat),math.radians(dlat); dlatr,dlonr=math.radians(dlat-plat),math.radians(dlon-plon)
    a=math.sin(dlatr/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlonr/2)**2
    hav=6371.0*2*math.asin(min(1,math.sqrt(a)))
    man=111*abs(dlat-plat)+111*abs(dlon-plon)*abs(math.cos(math.radians((plat+dlat)/2)))
    y=math.sin(dlonr)*math.cos(lat2); x=math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dlonr)
    bearing=(math.degrees(math.atan2(y,x))+360)%360
    row={'vendor_id':vendor,'passenger_count':passengers,'store_and_fwd_flag':flag,'pickup_longitude':plon,'pickup_latitude':plat,'dropoff_longitude':dlon,'dropoff_latitude':dlat,'dist_haversine':hav,'dist_manhattan':man,'log_dist':math.log1p(hav),'bearing':bearing,'hour_sin':math.sin(2*math.pi*hour/24),'hour_cos':math.cos(2*math.pi*hour/24),'dayofweek':dow,'month':month,'is_weekend':int(dow>=5),'is_rush_hour':int(hour in (7,8,9,16,17,18,19))}
    df=pd.DataFrame([row],columns=FEATURES)
    df['vendor_id']=pd.Categorical(df['vendor_id'],categories=[1,2])
    df['store_and_fwd_flag']=pd.Categorical(df['store_and_fwd_flag'],categories=['N','Y'])
    return df,hav,(plat,plon,dlat,dlon)

@app.get('/')
def home():
    return render_template_string(HTML,mae=f"{float(METRICS.get('mae_seconds',186.74))/60:.2f} min",rmse=f"{float(METRICS.get('rmse_seconds',298.35))/60:.2f} min",r2=f"{float(METRICS.get('r2',.7934)):.4f}")
@app.get('/health')
def health(): return jsonify(status='ok',model=bundle.get('model_name',type(model).__name__))
@app.post('/predict')
def predict():
    try:
        p=request.get_json(force=True); frame,dist,coords=build_features(p); seconds=max(0.0,float(model.predict(frame)[0])); a,b,c,d=coords
        return jsonify(seconds=seconds,minutes=seconds/60,distance_km=dist,pickup_lat=a,pickup_lon=b,dropoff_lat=c,dropoff_lon=d)
    except (KeyError,TypeError,ValueError) as e: return jsonify(error=str(e)),400
    except Exception:
        app.logger.exception('Prediction error'); return jsonify(error='Prediction failed. Please check the inputs and try again.'),500

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.environ.get('PORT','10000')),debug=False)
