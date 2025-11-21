# Import các thư viện cần thiết từ Flask và các thư viện khác
import sqlite3
from functools import wraps
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
# Đặt khóa bí mật (SECRET_KEY) cho Session. 
# Cần thiết cho các tính năng như session và flash.
app.secret_key = 'super_secret_key_for_faceswap_app'  # Nên thay bằng key mạnh hơn trong production

DATABASE = 'database.db'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo folder uploads nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- HÀM HỖ TRỢ ---

def get_db():
    """Tạo kết nối cơ sở dữ liệu và trả về đối tượng kết nối."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Cho phép truy cập cột bằng tên
    return conn

def init_db():
    """Khởi tạo cơ sở dữ liệu: Tạo bảng users nếu chưa tồn tại."""
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                avatar TEXT DEFAULT 'https://placehold.co/32x32/4caf50/white?text=U',
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        conn.commit()

# Khởi tạo DB khi ứng dụng bắt đầu
init_db()

def login_required(f):
    """Decorator để bảo vệ các tuyến đường, yêu cầu người dùng phải đăng nhập."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            flash('Bạn cần đăng nhập để truy cập trang này.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- TUYẾN ĐƯỜNG (ROUTES) ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Trang đăng nhập."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['logged_in'] = True
                session['email'] = email
                session['avatar'] = user['avatar']
                flash('Đăng nhập thành công!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Email hoặc mật khẩu không đúng.', 'error')
    
    return render_template('login.html')  # Giả sử bạn có template login.html

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Trang đăng ký."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if not email or not password or not confirm_password:
            flash('Vui lòng điền đầy đủ thông tin.', 'error')
        elif password != confirm_password:
            flash('Mật khẩu xác nhận không khớp.', 'error')
        elif len(password) < 6:
            flash('Mật khẩu phải có ít nhất 6 ký tự.', 'error')
        else:
            hashed_password = generate_password_hash(password)
            try:
                with get_db() as conn:
                    conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
                    conn.commit()
                flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Email đã tồn tại.', 'error')
    
    return render_template('register.html')  # Giả sử bạn có template register.html

@app.route('/')
@login_required  # Chỉ cho phép người dùng đã đăng nhập truy cập
def home():
    """Trang chủ của ứng dụng (Hiển thị giao diện hoán đổi khuôn mặt)."""
    return render_template('index.html', user_email=session.get('email'), avatar=session.get('avatar', ''))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """Trang hồ sơ: Hiển thị thông tin và cập nhật avatar/mật khẩu."""
    email = session['email']
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            flash('Không tìm thấy tài khoản.', 'error')
            return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Thay đổi avatar
        if 'avatar' in request.files:
            file = request.files['avatar']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{email.split('@')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                avatar_url = url_for('static', filename=f'uploads/{filename}')
                with get_db() as conn:
                    conn.execute("UPDATE users SET avatar = ? WHERE email = ?", (avatar_url, email))
                    conn.commit()
                session['avatar'] = avatar_url  # Cập nhật session
                flash('Cập nhật avatar thành công!', 'success')
                return redirect(url_for('profile'))
            else:
                flash('File không hợp lệ (chỉ JPG, PNG, JPEG, GIF).', 'error')
        
        # Đổi mật khẩu
        old_password = request.form.get('old_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if old_password and new_password and confirm_password:
            if new_password != confirm_password:
                flash('Mật khẩu mới và xác nhận không khớp.', 'error')
            elif not check_password_hash(user['password'], old_password):
                flash('Mật khẩu cũ không chính xác.', 'error')
            elif len(new_password) < 6:
                flash('Mật khẩu mới phải có ít nhất 6 ký tự.', 'error')
            else:
                hashed_password = generate_password_hash(new_password)
                with get_db() as conn:
                    conn.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
                    conn.commit()
                flash('Đổi mật khẩu thành công! Vui lòng đăng nhập lại.', 'success')
                session.clear()  # Force logout để áp dụng pass mới
                return redirect(url_for('login'))
        else:
            flash('Vui lòng điền đầy đủ thông tin.', 'error')
    
    # Hiển thị form (GET)
    created_date = user['created_date'].strftime('%d/%m/%Y') if user['created_date'] else 'N/A'
    return render_template('profile.html', user=user, created_date=created_date)

@app.route('/logout')
def logout():
    """Xử lý logic Đăng xuất."""
    session.clear()
    flash('Bạn đã đăng xuất thành công.', 'info')
    return redirect(url_for('login'))

# --- CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    # Chạy ứng dụng Flask. debug=True giúp tự động tải lại khi có thay đổi.
    app.run(debug=True)