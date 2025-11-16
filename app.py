from flask import Flask, render_template, request, redirect, session, jsonify
import hashlib

app.secret_key = "your_secret_key"

# ==========================
#  TẠO BẢNG USERS
# ==========================
def init_user_table():
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
        """)
        conn.commit()

# ==========================
#  LOGIN PAGE
# ==========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form["email"]
    password = hashlib.sha256(request.form["password"].encode()).hexdigest()

    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cur.fetchone()

    if user:
        session["user"] = email
        return redirect("/")

    return "Sai email hoặc mật khẩu!"

# ==========================
#  REGISTER PAGE
# ==========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    name = request.form["name"]
    email = request.form["email"]
    password = hashlib.sha256(request.form["password"].encode()).hexdigest()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (name, email, password)
                VALUES (%s, %s, %s)
            """, (name, email, password))
            conn.commit()
        return redirect("/login")

    except psycopg2.Error:
        return "Email đã tồn tại!"

# ==========================
#  LOGOUT
# ==========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")
