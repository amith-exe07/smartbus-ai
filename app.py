
import os
import math
import joblib
import pandas as pd

from flask import Flask, request, render_template_string, jsonify

# --------------------------------------------------
# APP AND MODEL SETUP
# --------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "smartbus_validated_model.joblib")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
METRICS = bundle.get("evaluation_reference", {})

app = Flask(__name__)

# --------------------------------------------------
# DASHBOARD HTML
# --------------------------------------------------

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>SmartBus AI | Trip Duration Estimator</title>

<link rel="stylesheet"
 href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #f2f6fc;
    color: #14233c;
    font-family: Arial, sans-serif;
}

header {
    background: #0c1f3d;
    color: white;
    padding: 24px max(20px, calc((100vw - 1120px) / 2));
}

header h1 {
    margin: 0 0 6px;
}

header p {
    margin: 0;
    color: #d1dced;
}

main {
    max-width: 1120px;
    margin: 24px auto;
    padding: 0 16px;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

.metric,
.panel {
    background: white;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 15px #102b4b0b;
}

.metric small {
    display: block;
    color: #61738b;
    margin-bottom: 8px;
}

.metric strong {
    font-size: 22px;
}

.cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

h2 {
    font-size: 20px;
    margin: 0 0 14px;
}

#map {
    height: 300px;
    border-radius: 10px;
}

.muted,
.mapnote {
    font-size: 12px;
    color: #61738b;
}

.mapnote {
    min-height: 22px;
    margin-top: 8px;
}

form {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

label {
    font-size: 13px;
    font-weight: bold;
    display: block;
    margin-bottom: 6px;
}

input,
select {
    width: 100%;
    padding: 10px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: white;
    font: inherit;
}

.full {
    grid-column: 1 / -1;
}

button {
    width: 100%;
    background: #2563eb;
    color: white;
    border: 0;
    border-radius: 9px;
    padding: 13px;
    font-size: 15px;
    font-weight: bold;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.result {
    margin-top: 15px;
    padding: 18px;
    border-radius: 12px;
    background: #eef5ff;
    overflow-wrap: anywhere;
}

.result small {
    font-weight: bold;
    color: #2563eb;
}

.result h3 {
    font-size: 29px;
    margin: 9px 0;
}

.result p {
    margin: 8px 0;
}

footer {
    max-width: 1120px;
    margin: auto;
    padding: 0 16px 25px;
    color: #64748b;
    font-size: 12px;
}

@media (max-width: 720px) {
    .metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .cols {
        grid-template-columns: 1fr;
    }
}
</style>
</head>

<body>

<header>
    <h1>🚌 SmartBus AI</h1>
    <p>AI-powered trip-duration estimation dashboard</p>
</header>

<main>

    <div class="metrics">
        <div class="metric">
            <small>Holdout MAE</small>
            <strong>{{ mae }}</strong>
        </div>

        <div class="metric">
            <small>Holdout RMSE</small>
            <strong>{{ rmse }}</strong>
        </div>

        <div class="metric">
            <small>Holdout R²</small>
            <strong>{{ r2 }}</strong>
        </div>

        <div class="metric">
            <small>Model inputs</small>
            <strong>17 features</strong>
        </div>
    </div>

    <div class="cols">

        <section class="panel">
            <h2>Choose a route</h2>

            <div id="map"></div>

            <div class="mapnote" id="mapnote">
                Click the map once for pickup, then again for drop-off.
            </div>

            <p class="muted">
                Map centred on New York City.
            </p>
        </section>

        <section class="panel">
            <h2>Trip details</h2>

            <form id="form">

                <div>
                    <label>Pickup latitude</label>
                    <input name="pickup_lat" type="number"
                        step="any" value="40.75958" required>
                </div>

                <div>
                    <label>Pickup longitude</label>
                    <input name="pickup_lon" type="number"
                        step="any" value="-74.00296" required>
                </div>

                <div>
                    <label>Drop-off latitude</label>
                    <input name="dropoff_lat" type="number"
                        step="any" value="40.77112" required>
                </div>

                <div>
                    <label>Drop-off longitude</label>
                    <input name="dropoff_lon" type="number"
                        step="any" value="-73.99361" required>
                </div>

                <div>
                    <label>Passengers</label>
                    <select name="passenger_count">
                        {% for n in range(1, 7) %}
                        <option value="{{ n }}">{{ n }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label>Vendor</label>
                    <select name="vendor_id">
                        <option value="1">Vendor 1</option>
                        <option value="2">Vendor 2</option>
                    </select>
                </div>

                <div>
                    <label>Store-and-forward flag</label>
                    <select name="store_and_fwd_flag">
                        <option value="N">N</option>
                        <option value="Y">Y</option>
                    </select>
                </div>

                <div>
                    <label>Pickup hour (0–23)</label>
                    <input name="hour" type="number"
                        min="0" max="23" value="17" required>
                </div>

                <div>
                    <label>Day of week</label>
                    <select name="dayofweek">
                        {% for n in range(7) %}
                        <option value="{{ n }}"
                            {% if n == 2 %}selected{% endif %}>
                            {{ ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][n] }}
                        </option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label>Month (1–12)</label>
                    <input name="month" type="number"
                        min="1" max="12" value="6" required>
                </div>

                <div class="full">
                    <button type="submit">
                        Predict Trip Duration
                    </button>
                </div>

            </form>

            <div class="result" id="result">
                <small>ESTIMATED TRIP DURATION</small>
                <h3>Ready to predict</h3>
                <p>Enter route details and click Predict.</p>
            </div>

        </section>
    </div>

</main>

<footer>
    SmartBus AI · Student trip-duration estimation project
    · Evaluation metrics are from the model's holdout dataset.
</footer>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const map = L.map("map").setView([40.75, -73.99], 12);

L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap"
    }
).addTo(map);

let pickupMarker = null;
let dropoffMarker = null;
let routeLine = null;
let clickCount = 0;

map.on("click", function (event) {
    const form = document.getElementById("form");
    const lat = event.latlng.lat.toFixed(5);
    const lon = event.latlng.lng.toFixed(5);

    if (clickCount % 2 === 0) {
        if (pickupMarker) {
            map.removeLayer(pickupMarker);
        }

        pickupMarker = L.marker(event.latlng)
            .addTo(map)
            .bindPopup("Pickup")
            .openPopup();

        form.elements.pickup_lat.value = lat;
        form.elements.pickup_lon.value = lon;

        document.getElementById("mapnote").textContent =
            "Pickup selected. Click again for drop-off.";
    } else {
        if (dropoffMarker) {
            map.removeLayer(dropoffMarker);
        }

        if (routeLine) {
            map.removeLayer(routeLine);
        }

        dropoffMarker = L.marker(event.latlng)
            .addTo(map)
            .bindPopup("Drop-off")
            .openPopup();

        form.elements.dropoff_lat.value = lat;
        form.elements.dropoff_lon.value = lon;

        if (pickupMarker) {
            routeLine = L.polyline(
                [pickupMarker.getLatLng(), event.latlng],
                { color: "#2563eb" }
            ).addTo(map);
        }

        document.getElementById("mapnote").textContent =
            "Route selected. Adjust details and predict.";
    }

    clickCount++;
});

document.getElementById("form").addEventListener(
    "submit",
    async function (event) {
        event.preventDefault();

        const output = document.getElementById("result");

        output.innerHTML = "<p>Calculating...</p>";

        const payload = Object.fromEntries(
            new FormData(event.target).entries()
        );

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed");
            }

            output.replaceChildren();

            const heading = document.createElement("small");
            heading.textContent = "ESTIMATED TRIP DURATION";

            const duration = document.createElement("h3");
            duration.textContent =
                data.minutes.toFixed(2) + " minutes";

            const seconds = document.createElement("p");
            seconds.textContent =
                "Approximately " + Math.round(data.seconds) + " seconds";

            const distance = document.createElement("p");
            distance.textContent =
                "Straight-line distance: " +
                data.distance_km.toFixed(3) + " km";

            const pickup = document.createElement("p");
            pickup.textContent =
                "Pickup: " + data.pickup_lat.toFixed(5) +
                ", " + data.pickup_lon.toFixed(5);

            const dropoff = document.createElement("p");
            dropoff.textContent =
                "Drop-off: " + data.dropoff_lat.toFixed(5) +
                ", " + data.dropoff_lon.toFixed(5);

            const note = document.createElement("p");
            note.className = "muted";
            note.textContent =
                "Estimated duration; not a guaranteed road travel time.";

            output.append(
                heading, duration, seconds, distance,
                pickup, dropoff, note
            );

        } catch (error) {
            output.replaceChildren();

            const message = document.createElement("p");
            const strong = document.createElement("b");

            strong.textContent = "Could not predict: ";
            message.append(strong, document.createTextNode(error.message));

            output.appendChild(message);
        }
    }
);
</script>

</body>
</html>
"""

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def build_features(p):
    plat = float(p["pickup_lat"])
    plon = float(p["pickup_lon"])
    dlat = float(p["dropoff_lat"])
    dlon = float(p["dropoff_lon"])

    passengers = int(p.get("passenger_count", 1))
    vendor = int(p.get("vendor_id", 1))

    flag = str(p.get("store_and_fwd_flag", "N")).upper()
    hour = int(p.get("hour", 17))
    dow = int(p.get("dayofweek", 2))
    month = int(p.get("month", 6))

    for name, value, low, high in [
        ("pickup_lat", plat, -90, 90),
        ("dropoff_lat", dlat, -90, 90),
        ("pickup_lon", plon, -180, 180),
        ("dropoff_lon", dlon, -180, 180),
    ]:
        if not low <= value <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}."
            )

    if not 1 <= passengers <= 6:
        raise ValueError("Passenger count must be from 1 to 6.")

    if vendor not in (1, 2):
        raise ValueError("Vendor must be 1 or 2.")

    if flag not in ("N", "Y"):
        raise ValueError("Store-and-forward flag must be N or Y.")

    if not 0 <= hour <= 23:
        raise ValueError("Hour must be from 0 to 23.")

    if not 0 <= dow <= 6:
        raise ValueError("Day of week must be from 0 to 6.")

    if not 1 <= month <= 12:
        raise ValueError("Month must be from 1 to 12.")

    lat1 = math.radians(plat)
    lat2 = math.radians(dlat)

    delta_lat = math.radians(dlat - plat)
    delta_lon = math.radians(dlon - plon)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    haversine = 6371.0 * 2 * math.asin(
        min(1, math.sqrt(a))
    )

    manhattan = (
        111 * abs(dlat - plat)
        + 111 * abs(dlon - plon)
        * abs(math.cos(math.radians((plat + dlat) / 2)))
    )

    y = math.sin(delta_lon) * math.cos(lat2)

    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2)
        * math.cos(delta_lon)
    )

    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    row = {
        "vendor_id": vendor,
        "passenger_count": passengers,
        "store_and_fwd_flag": flag,
        "pickup_longitude": plon,
        "pickup_latitude": plat,
        "dropoff_longitude": dlon,
        "dropoff_latitude": dlat,
        "dist_haversine": haversine,
        "dist_manhattan": manhattan,
        "log_dist": math.log1p(haversine),
        "bearing": bearing,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dayofweek": dow,
        "month": month,
        "is_weekend": int(dow >= 5),
        "is_rush_hour": int(hour in (7, 8, 9, 16, 17, 18, 19)),
    }

    frame = pd.DataFrame([row], columns=FEATURES)

    frame["vendor_id"] = pd.Categorical(
        frame["vendor_id"], categories=[1, 2]
    )

    frame["store_and_fwd_flag"] = pd.Categorical(
        frame["store_and_fwd_flag"], categories=["N", "Y"]
    )

    return frame, haversine, (plat, plon, dlat, dlon)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.get("/")
def home():
    return render_template_string(
        HTML,
        mae=f"{float(METRICS.get('mae_seconds', 186.74)) / 60:.2f} min",
        rmse=f"{float(METRICS.get('rmse_seconds', 298.35)) / 60:.2f} min",
        r2=f"{float(METRICS.get('r2', 0.7934)):.4f}",
    )


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model=bundle.get("model_name", type(model).__name__)
    )


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)

        frame, distance, coordinates = build_features(payload)

        seconds = max(
            0.0,
            float(model.predict(frame)[0])
        )

        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = coordinates

        return jsonify(
            seconds=seconds,
            minutes=seconds / 60,
            distance_km=distance,
            pickup_lat=pickup_lat,
            pickup_lon=pickup_lon,
            dropoff_lat=dropoff_lat,
            dropoff_lon=dropoff_lon,
        )

    except (KeyError, TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400

    except Exception:
        app.logger.exception("Prediction error")

        return jsonify(
            error="Prediction failed. Please check the inputs and try again."
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False,
    )
