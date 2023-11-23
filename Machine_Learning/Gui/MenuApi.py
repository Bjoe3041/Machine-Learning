import threading

import numpy as np
from flask import Flask, request, jsonify
import tkinter as tk
from Controller.MLController import MLController


def openapi():
    app = Flask(__name__)
    print("apiview opened")
    textConsole.insert(tk.END, "API STARTED")

    @app.route('/api/status', methods=['GET'])
    def get_status():
        textConsole.insert(tk.END, "\n" + "STATUS REQUEST")
        return jsonify({"status": "Server is up"})

    @app.route('/api/predict', methods=['POST'])
    def convert_string_to_integer():
        data = request.get_json()  # Assumes the client sends JSON data
        title = data['title']  # Assuming the client sends a JSON object with a key 'input_string'

        result = ml_getvalue(title)

        return jsonify({"result": result})

    @app.route('/api/predict/compare', methods=['POST'])
    def compare_titles():
        data = request.get_json()  # Assumes the client sends JSON data
        title1 = data['title1']  # Assuming the client sends a JSON object with a key 'input_string'
        title2 = data['title2']  # Assuming the client sends a JSON object with a key 'input_string'

        result = ml_getpercentages(title1, title2)

        return jsonify({"result": result})

    app.run()


def topercent(val1, val2):
    valsum = val1 + val2
    x = 1 / valsum
    return [val1 * x, val2 * x]


def ml_getpercentages(input1, input2):
    mlc = MLController()
    eval1 = mlc.evaluatetitle(input1)
    eval2 = mlc.evaluatetitle(input2)

    p = topercent(eval1, eval2)

    returnobj = {
        "title1": [f"{p[0]*100:.3f}%", input1],
        "title2": [f"{p[1]*100:.3f}%", input2],
        "delta": f"{np.abs(p[1] - p[0])*100:.3f}%"
     }

    return returnobj


def ml_getvalue(inputtitle):
    mlc = MLController()
    retvalue = mlc.evaluatetitle(inputtitle)
    textConsole.insert(tk.END, "\n" + "EVAL REQUEST - [" + inputtitle + "][" + f"{retvalue:.3f}" + "]")
    return retvalue


def start_api_thread():
    api_thread = threading.Thread(target=openapi)
    api_thread.daemon = True  # Set the thread as a daemon, so it will exit when the main program exits
    api_thread.start()


newwindow = tk.Tk()
newwindow.title("ML Api Window")
newwindow.geometry("450x560")
newwindow.config(bg='#1c1f21')

console_text_color = "#6dc0d1"
console_bg_color = "#201924"

# Create label
labelTitle = tk.Label(newwindow, text="API SERVERSIDE")
labelTitle.config(font=("Helvetica", 14), bg=console_bg_color, fg=console_text_color)

textConsole = tk.Text(newwindow,
                      highlightcolor='#6dc0d1',
                      bg=console_bg_color,
                      borderwidth=4,
                      fg=console_text_color,
                      font='Helvetica')

b1 = tk.Button(newwindow,
               bg=console_text_color,
               fg=console_bg_color,
               width=50,
               height=1,
               text="HOST API",
               command=start_api_thread)

b2 = tk.Button(newwindow,
               bg=console_text_color,
               fg=console_bg_color,
               width=50,
               height=1,
               text="Exit",
               command=newwindow.destroy)

padding_frame = tk.Frame(newwindow, height=20, bg='#1c1f21')

# Configure grid for widgets
labelTitle.grid(row=0, column=0, columnspan=2)
textConsole.grid(row=1, column=0, columnspan=2, sticky="nsew")  # Add sticky to make it expand
b1.grid(row=2, column=0, padx=10, pady=10)
b2.grid(row=2, column=1, padx=10, pady=10)
padding_frame.grid(row=3, column=0, columnspan=2)

# Configure column weights to make both columns expand with window width
newwindow.columnconfigure(0, weight=1)
newwindow.columnconfigure(1, weight=1)

# Configure row weight to make the Text widget expand vertically
newwindow.rowconfigure(1, weight=1)

newwindow.mainloop()
