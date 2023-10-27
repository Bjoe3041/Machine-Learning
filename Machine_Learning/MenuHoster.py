import tkinter as tk
from flask import Flask, request, jsonify
import os
import sys
import threading

class MenuHoster:
    def hostmenu(self):
        root = tk.Tk()
        root.title("Machine Learning Menu")
        window_size = 400
        root.geometry(f"{window_size}x{window_size}")
        menu = tk.Menu(root)
        root.config(menu=menu)
        submenu = tk.Menu(menu)
        menu.add_cascade(label="Functions", menu=submenu)
        submenu.add_command(label="Smth i dunno yet", command=self.function1)
        submenu.add_command(label="Smth i dunno yet", command=self.function2)
        button1 = tk.Button(root, width=50, text="Trial run", command=self.open_trialview)
        button2 = tk.Button(root, width=50, text="Accuracy", command=self.open_accuracyview)
        button3 = tk.Button(root, width=50, text="Host as api", command=self.open_apiview)
        button1.pack()
        button2.pack()
        button3.pack()
        root.mainloop()

    def open_trialview(self):
        # Add your code here for the first function
        pass

    # Function to be called when Button 2 is clicked
    def open_accuracyview(self):
        # Add your code here for the second function
        pass

    def thrd_open_apiview(self):
        os.system(f"{sys.executable} ./api_server.py")
        print("apiview clicked")

    def open_apiview(self):
        apiview_thread = threading.Thread(target=self.thrd_open_apiview)
        apiview_thread.daemon = True  # Set the thread as a daemon so it will exit when the main program exits
        apiview_thread.start()


    def get_status(self):
        return "Gamign123 we up here bois"


    def function1(self):
        # Add your code here for the first function
        pass


    def function2(self):
        # Add your code here for the first function
        pass


if __name__ == '__main__':
    menu_hoster = MenuHoster()
    menu_hoster.hostmenu()