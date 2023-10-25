import threading
from flask import Flask, jsonify
import tkinter as tk

def openapi():
    app = Flask(__name__)
    print("apiview opened")
    textConsole.insert(tk.END, "\n" + "API STARTED")

    @app.route('/api/status', methods=['GET'])
    def get_status():
        textConsole.insert(tk.END, "\n" + "STATUS REQUEST")
        return jsonify({"status": "Gaming123, we up here bois"})

    @app.route('/api/check', methods=['POST'])
    def convert_string_to_integer():
        data = request.get_json()  # Assumes the client sends JSON data
        title = data['title']  # Assuming the client sends a JSON object with a key 'input_string'

        result = ml_getvalue()

        return jsonify({"result": result})

    app.run()

def ml_getvalue():
    pass


def start_api_thread():
    api_thread = threading.Thread(target=openapi)
    api_thread.daemon = True  # Set the thread as a daemon so it will exit when the main program exits
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

frame = tk.Frame(newwindow, padx=1, pady=1, bg='#1c1f21') # i added this frame so i can pad the text

textConsole = tk.Text(frame, highlightcolor='#6dc0d1', bg=console_bg_color, borderwidth=4, fg=console_text_color, font='Helvetica', width=45)
b1 = tk.Button(newwindow, bg = console_text_color, fg = console_bg_color, width=50, height = 1, text="HOST API", command=start_api_thread)
b2 = tk.Button(newwindow, bg = console_text_color, fg = console_bg_color, width=50, height = 1, text="Exit", command=newwindow.destroy)


padding_frame = tk.Frame(newwindow, height=20, bg='#1c1f21')

labelTitle.pack()
frame.pack(fill=tk.BOTH, expand=True)
textConsole.pack()
b1.pack(pady=10)
b2.pack(pady=1)
padding_frame.pack()

newwindow.mainloop()
