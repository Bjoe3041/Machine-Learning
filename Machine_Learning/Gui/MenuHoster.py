import tkinter as tk
import os
import sys
import threading
from Controller.MLController import MLController


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
        submenu.add_command(label="Retrain", command=self.retrain)
        submenu.add_command(label="Smth i dunno yet", command=self.quicktest)
        button_trial = tk.Button(root, width=50, text="Trial run", command=self.open_trialview)
        button_accuracy = tk.Button(root, width=50, text="Accuracy", command=self.open_accuracyview)
        button_api = tk.Button(root, width=50, text="Host as api", command=self.open_apiview)
        button_trial.pack()
        button_accuracy.pack()
        button_api.pack()
        root.mainloop()

    def open_trialview(self):
        pass

    def open_accuracyview(self):
        pass

    @staticmethod
    def thrd_open_apiview():
        current_directory = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_directory, "api_server.py")
        os.system(f"{sys.executable} {script_path}")
        # os.system(f"{sys.executable} api_server.py")

    def open_apiview(self):
        apiview_thread = threading.Thread(target=self.thrd_open_apiview)
        apiview_thread.daemon = True  # Set the thread as a daemon, so it will exit when the main program exits
        apiview_thread.start()

    @staticmethod
    def retrain():
        mlc = MLController()
        modelname = "model_test1_gaming"
        vectorizername = "vector_test1_gaming"

        pth = os.getcwd()
        print(os.listdir(pth))

        model, vectorizer = mlc.trainmodel('Machine_Learning/Corrected_2_Updated_Preferred_titles.xlsx')
        mlc.savemodel(model, vectorizer, modelname, vectorizername)

    @staticmethod
    def quicktest():
        mlc = MLController()
        modelname = "model_test1_gaming"
        vectorizername = "vector_test1_gaming"
        # mlc.savemodel(model, vectorizer, modelname, vectorizername)
        loadedmodel, loadedvectorizer = mlc.loadmodel(modelname, vectorizername)

        print(mlc.evaluate(loadedmodel, loadedvectorizer, "role"))


if __name__ == '__main__':
    menu_hoster = MenuHoster()
    menu_hoster.hostmenu()
