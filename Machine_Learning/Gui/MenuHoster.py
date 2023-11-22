import tkinter as tk
import os
import sys
import threading
from Controller.MLController import MLController


def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found"


def update_label_content(label):
    file_path = 'Machine_Learning/modelpath.txt'  # Replace with your file path
    content = read_file_content(file_path)
    label.config(text=content)


def getoptions():
    reallist = []
    # print(os.getcwd())

    file_list = os.listdir(os.getcwd() + '\\Machine_Learning\\Saves')
    for i in range(0, len(file_list)):
        name = file_list[i].removeprefix("model_")
        if "vector_" not in name:
            # print(name)
            reallist.append(name)
    return reallist


class MenuHoster:
    def hostmenu(self):
        def refresh_options():
            new_options = getoptions()
            dropdown_var.set(MLController().getchosenmodelpath()) # Todo, set to current selected file from folder instead
            dropdown_menu['menu'].delete(0, 'end')
            for option in new_options:
                dropdown_menu['menu'].add_command(label=option, command=lambda opt=option: set_modelname(opt))

        def set_modelname(content):
            print("name SET")
            mlc = MLController()
            mlc.setchosenmodelpath(content)
            update_label_content(label)

        root = tk.Tk()
        root.title("Machine Learning Menu")
        window_size = 400
        root.geometry(f"{window_size}x{window_size}")
        menu = tk.Menu(root)
        root.config(menu=menu)
        submenu = tk.Menu(menu)
        menu.add_cascade(label="Functions", menu=submenu)
        submenu.add_command(label="Retrain - (deprecated)", command=self.retrain)
        submenu.add_command(label="Quicktest to console", command=self.quicktest)
        button_trial = tk.Button(root, width=50, text="Trial run", command=self.open_trialview)
        button_accuracy = tk.Button(root, width=50, text="Accuracy", command=self.open_accuracyview)
        button_api = tk.Button(root, width=50, text="Host as api", command=self.open_apiview)
        button_train = tk.Button(root, width=50, text="Train model again", command=self.open_trainview)

        button_frame = tk.Frame(root, width=359, height=20)
        button_frame.pack_propagate(False)
        label = tk.Label(button_frame, text="", wraplength=400)  # Adjust wraplength as needed

        options = getoptions()
        dropdown_var = tk.StringVar()
        options.append("None Selected")
        dropdown_var.set(MLController().getchosenmodelpath())

        dropdown_menu = tk.OptionMenu(root, dropdown_var, *options)
        dropdown_menu.config(width=53)


        update_button = tk.Button(button_frame, text="⭮", command=refresh_options)
        update_label_content(label)

        label_modeldesc = tk.Label(button_frame, text="", wraplength=400)  # Adjust wraplength as needed
        label_modeldesc.config(text="Selected Model:")

        button_trial.pack()
        button_accuracy.pack()
        button_api.pack()
        button_train.pack()
        dropdown_menu.pack(padx=0, pady=(120, 0))
        button_frame.pack(side="top")

        label_modeldesc.pack(side="left", padx=0, pady=0, anchor='w')
        label.pack(side="left", padx=5)
        update_button.pack(side="right")

        def on_option_menu_map(event):
            # This function is triggered when the OptionMenu is mapped to the screen
            refresh_options()

        dropdown_menu.bind("<Enter>", on_option_menu_map) ##Sometimes runs in wacky order, be careful


        # Some fun stuff
        # nerd_emoji = "\U0001F913"  # Unicode for Nerd Face emoji
        # label = tk.Label(root, text=nerd_emoji, font=("Arial", 10), fg='#24242b', borderwidth=4, anchor='s')
        # label.pack()
        # some fun stuff

        root.mainloop()

    def open_trialview(self):
        pass

    def open_accuracyview(self):
        pass

    @staticmethod
    def thrd_open_apiview():
        current_directory = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_directory, "MenuApi.py")
        print(script_path)
        os.system(f"{sys.executable} \"{script_path}\"")

    def open_apiview(self):
        apiview_thread = threading.Thread(target=self.thrd_open_apiview)
        apiview_thread.daemon = True  # Set the thread as a daemon, so it will exit when the main program exits
        apiview_thread.start()

    @staticmethod
    def thrd_open_trainview():
        current_directory = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_directory, "MenuTrain.py")
        os.system(f"{sys.executable} {script_path}")

    def open_trainview(self):
        trainview_thread = threading.Thread(target=self.thrd_open_trainview)
        trainview_thread.daemon = True
        trainview_thread.start()

    @staticmethod
    def retrain():
        mlc = MLController()
        name = mlc.getchosenmodelpath()

        modelname = "model_" + name
        vectorizername = "vector_" + name
        model, vectorizer = mlc.trainmodel_excel('Machine_Learning/Corrected_2_Updated_Preferred_titles.xlsx')
        mlc.savemodel(model, vectorizer, modelname, vectorizername)

    @staticmethod
    def quicktest():
        mlc = MLController()
        print(mlc.evaluate("role"))


if __name__ == '__main__':
    menu_hoster = MenuHoster()
    menu_hoster.hostmenu()
