from flask import Flask
app = Flask(__name__)
@app.route("/")
def Greeting():
    return "DS620 Emissions API"
if __name__ == "__main__":
    app.run()