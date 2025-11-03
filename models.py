from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username}>"

class VideoUpload(db.Model):
    __tablename__ = "video_uploads"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    user = db.relationship("User", backref=db.backref("videos", lazy=True))

    def __repr__(self):
        return f"<Video {self.filename}>"

class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey("video_uploads.id"), nullable=False)
    label = db.Column(db.String(10))  # "FAKE" or "REAL"
    confidence = db.Column(db.Float)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)

    video = db.relationship("VideoUpload", backref=db.backref("predictions", lazy=True))

    def __repr__(self):
        return f"<Prediction {self.label} ({self.confidence})>"
