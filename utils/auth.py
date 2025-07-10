from flask import session

def check_login(username, password):
    return username == "doctor" and password == "securepass"

def login_user(username):
    session['user'] = username

def is_logged_in():
    return 'user' in session

def logout_user():
    session.pop('user', None)
