# app.py

from database import init_db, fix_database
from flask import Flask, render_template, request, redirect, session, send_file, jsonify
import numpy as np
import sqlite3
import io
import pickle
import re
import pdfplumber

from tensorflow.keras.models import load_model

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ======================================================
# APP CONFIG
# ======================================================
app = Flask(__name__)
app.secret_key = "secret123"

init_db()
fix_database()

# ======================================================
# LOAD MODEL
# ======================================================
model = load_model("models/heart_model.h5")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

SEX_MAP = {
    1: "Male",
    0: "Female"
}

CP_MAP = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}

EXANG_MAP = {
    1: "Yes",
    0: "No"
}

THAL_MAP = {
    0: "Unknown",
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect"
}

SLOPE_MAP = {
    0: "Downsloping",
    1: "Flat",
    2: "Upsloping"
}

RESTECG_MAP = {
    0: "Normal",
    1: "ST-T Wave Abnormality",
    2: "Left Ventricular Hypertrophy"
}

# ======================================================
# DATABASE
# ======================================================
def get_db():
    return sqlite3.connect("patients.db")


# ======================================================
# LOGIN
# ======================================================
@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/home")

        return render_template(
            "login.html",
            error="Invalid Credentials"
        )

    return render_template("login.html")


# ======================================================
# REGISTER
# ======================================================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()

        try:
            cur.execute(
                "INSERT INTO users(username,password) VALUES(?,?)",
                (username, password)
            )
            conn.commit()
            conn.close()
            return redirect("/")

        except:
            conn.close()
            return render_template(
                "register.html",
                error="User already exists"
            )

    return render_template("register.html")


# ======================================================
# HOME
# ======================================================
@app.route("/home")
def home():

    if "user" not in session:
        return redirect("/")

    return render_template("index.html")


# ======================================================
# DASHBOARD
# ======================================================
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/")

    conn = get_db()
    cur = conn.cursor()

    # Get total predictions
    cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE username=?",
        (session["user"],)
    )
    total = cur.fetchone()[0]

    # Get risk cases
    cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE username=? AND result=?",
        (session["user"], "Risk")
    )
    risk = cur.fetchone()[0]

    # Get safe cases
    safe = total - risk

    conn.close()

    return render_template("dashboard.html", total=total, risk=risk, safe=safe)


# ======================================================
# PDF AUTOFILL
# ======================================================
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():

    if "user" not in session:
        return jsonify({"error": "Login required"}), 401

    file = request.files["pdf"]
    text = ""

    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    def get_num(pattern):
        m = re.search(pattern, text, re.I)
        return m.group(1) if m else ""

    data = {}

    data["age"] = get_num(r"Age\s+(\d+)")
    data["trestbps"] = get_num(r"Blood Pressure\s+(\d+)")
    data["chol"] = get_num(r"Cholesterol\s+(\d+)")
    data["thalach"] = get_num(r"Heart Rate\s+(\d+)")
    data["oldpeak"] = get_num(r"Oldpeak\s+([\d.]+)")
    data["ca"] = get_num(r"Major Vessels\s+(\d+)")

    data["sex"] = "1" if "Sex Male" in text else "0"
    data["fbs"] = "1" if "True" in text else "0"
    data["exang"] = "1" if "Exercise Angina Yes" in text else "0"

    if "Typical Angina" in text:
        data["cp"] = "0"
    elif "Atypical Angina" in text:
        data["cp"] = "1"
    elif "Non-anginal Pain" in text:
        data["cp"] = "2"
    else:
        data["cp"] = "3"

    if "Rest ECG Normal" in text:
        data["restecg"] = "0"
    elif "ST-T" in text:
        data["restecg"] = "1"
    else:
        data["restecg"] = "2"

    if "Downsloping" in text:
        data["slope"] = "0"
    elif "Flat" in text:
        data["slope"] = "1"
    else:
        data["slope"] = "2"

    if "Thal Normal" in text:
        data["thal"] = "1"
    elif "Fixed Defect" in text:
        data["thal"] = "2"
    elif "Reversible Defect" in text:
        data["thal"] = "3"
    else:
        data["thal"] = "0"

    return jsonify(data)


# ======================================================
# PREDICT
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect("/")

    values = [float(request.form[x]) for x in FEATURES]

    arr = np.array([values])
    arr = scaler.transform(arr)

    prob = float(model.predict(arr, verbose=0)[0][0])

    pred = 1 if prob >= 0.45 else 0

    result = "Risk" if pred == 1 else "Safe"

    confidence = round(prob * 100, 2) if pred == 1 else round((1-prob) * 100, 2)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO predictions(
    username,age,sex,cp,trestbps,chol,fbs,
    restecg,thalach,exang,oldpeak,slope,
    ca,thal,result,confidence
    )
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session["user"],
        *values,
        result,
        confidence
    ))

    conn.commit()
    
    cur.execute("SELECT last_insert_rowid()")
    prediction_id = cur.fetchone()[0]
    
    conn.close()

    # Return JSON for AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "prediction_id": prediction_id
        })
    
    # Return HTML for regular form submissions
    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


# ======================================================
# HISTORY
# ======================================================
@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, age, result, confidence, created_at
    FROM predictions
    WHERE username=?
    ORDER BY id DESC
    """, (session["user"],))

    rows = cur.fetchall()
    conn.close()

    return render_template("history.html", records=rows)


# ======================================================
# PREMIUM PDF REPORT
# ======================================================
@app.route("/download_report/<int:prediction_id>")
def download_report(prediction_id):

    if "user" not in session:
        return redirect("/")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT age,sex,cp,trestbps,chol,fbs,restecg,
           thalach,exang,oldpeak,slope,ca,thal,
           result,confidence,created_at
    FROM predictions
    WHERE id=? AND username=?
    """, (prediction_id, session["user"]))

    row = cur.fetchone()
    conn.close()

    if not row:
        return "Report Not Found"

    (
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal,
        result, confidence, created_at
    ) = row

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25,
        leftMargin=25,
        topMargin=20,
        bottomMargin=20
    )

    styles = getSampleStyleSheet()
    story = []

    # ======================================================
    # HEADER
    # ======================================================
    header = Table(
        [["HEART DISEASE CLINICAL REPORT"]],
        colWidths=[540]
    )

    header.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 20),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12)
    ]))

    story.append(header)
    story.append(Spacer(1, 15))

    # ======================================================
    # REPORT INFO
    # ======================================================
    story.append(Paragraph("<b>Report Information</b>", styles["Heading2"]))

    info = Table([
        ["User", session["user"]],
        ["Date", str(created_at).split(" ")[0]],
        ["ID", str(prediction_id)]
    ], colWidths=[170, 370])

    info.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.lightgrey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#dbeafe"))
    ]))

    story.append(info)
    story.append(Spacer(1, 14))

    # ======================================================
    # PATIENT DETAILS
    # ======================================================
    story.append(Paragraph("<b>Patient Details</b>", styles["Heading2"]))

    patient = [
        ["Age", int(age)],
        ["Sex", SEX_MAP[int(sex)]],
        ["Chest Pain", CP_MAP[int(cp)]],
        ["Blood Pressure", int(trestbps)],
        ["Cholesterol", int(chol)],
        ["Fasting Blood Sugar", "True" if int(fbs)==1 else "False"],
        ["Rest ECG", RESTECG_MAP[int(restecg)]],
        ["Heart Rate", int(thalach)],
        ["Exercise Angina", EXANG_MAP[int(exang)]],
        ["Oldpeak", oldpeak],
        ["Slope", SLOPE_MAP[int(slope)]],
        ["Major Vessels", int(ca)],
        ["Thal", THAL_MAP[int(thal)]]
    ]

    patient_table = Table(patient, colWidths=[220, 320])

    patient_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.lightgrey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eff6ff"))
    ]))

    story.append(patient_table)
    story.append(Spacer(1, 15))

    # ======================================================
    # RESULT
    # ======================================================
    story.append(Paragraph("<b>Prediction Result</b>", styles["Heading2"]))

    status_color = "#ef4444" if result == "Risk" else "#16a34a"

    result_box = Table([
        [f"{result} ({confidence}%)"]
    ], colWidths=[540])

    result_box.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor(status_color)),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 14),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10)
    ]))

    story.append(result_box)
    story.append(Spacer(1, 15))

    # ======================================================
    # RECOMMENDATION
    # ======================================================
    story.append(Paragraph("<b>Recommendation</b>", styles["Heading2"]))

    if result == "Risk":
        rec = "Consult a cardiologist for further evaluation and follow-up."
    else:
        rec = "Maintain healthy lifestyle and continue regular checkups."

    story.append(Paragraph(rec, styles["Normal"]))
    story.append(Spacer(1, 20))

    # ======================================================
    # DISCLAIMER
    # ======================================================
    disclaimer = ParagraphStyle(
        "small",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey
    )

    story.append(Paragraph(
        "Disclaimer: This report is generated for academic purposes only "
        "and should not replace professional medical advice.",
        disclaimer
    ))

    doc.build(story)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"report_{prediction_id}.pdf",
        mimetype="application/pdf"
    )


# ======================================================
# LOGOUT
# ======================================================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
