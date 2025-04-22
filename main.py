# Импортируем Flask
from flask import Flask, render_template

# Создаём экземпляр приложения Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("StartPage.html")

@app.route("/start")
def start():
    return render_template("DiscrablePage.html")

@app.route("/voice")
def voice():
    return render_template("MainPage.html")

# Запускаем сервер
if __name__ == "__main__":
    app.run(debug=True)  # debug=True включает автоматическую перезагрузку