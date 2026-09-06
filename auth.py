import hashlib
from database import db


class AuthManager:

    @staticmethod
    def hash_password(password: str):
        return hashlib.sha256(password.encode()).hexdigest()

    # ---------------------------
    # Signup
    # ---------------------------
    def signup(self, name, email, password, role):
        try:
            db.cursor.execute(
                """
                INSERT INTO users (name, email, password, role)
                VALUES (?, ?, ?, ?)
                """,
                (
                    name.strip(),
                    email.strip(),
                    self.hash_password(password),
                    role
                )
            )

            db.conn.commit()
            return True

        except Exception:
            return False

    # ---------------------------
    # Login
    # ---------------------------
    def login(self, email, password):

        db.cursor.execute(
            """
            SELECT *
            FROM users
            WHERE email = ? AND password = ?
            """,
            (
                email.strip(),
                self.hash_password(password)
            )
        )

        return db.cursor.fetchone()

    # ---------------------------
    # Forgot Password
    # ---------------------------
    def reset_password(self, email, new_password):

        email = email.strip()

        # Check whether email exists
        db.cursor.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        )

        user = db.cursor.fetchone()

        if not user:
            return False

        # Hash new password
        hashed_password = self.hash_password(new_password)

        # Update password
        db.cursor.execute(
            """
            UPDATE users
            SET password = ?
            WHERE email = ?
            """,
            (
                hashed_password,
                email
            )
        )

        db.conn.commit()

        return True