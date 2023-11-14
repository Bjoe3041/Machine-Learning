import os
import tkinter as tk
from Controller.MLController import MLController

newwindow = tk.Tk()
newwindow.title("ML training window")
# newwindow.geometry("450x560")
newwindow.config(bg='#1c1f21')

console_text_color = "#6dc0d1"
console_bg_color = "#201924"

# Create label
labelTitle = tk.Label(newwindow, text="TRAINING MONTAGE")
labelTitle.config(font=("Helvetica", 14), bg=console_bg_color, fg=console_text_color)


# modelnametextbox = tk.Text(newwindow)
# modelnametextbox.pack()

def on_entry_click(event):
    """Function that gets called whenever entry is clicked."""
    if entry.get() == 'Modelname':
        entry.delete(0, tk.END)  # delete all the text in the entry
        entry.insert(0, '')  # insert blank for user input
        entry.config(fg='#c68be8')


def on_focusout(event):
    """Function that gets called when the user leaves the entry widget."""
    if entry.get() == '':
        entry.insert(0, 'Modelname')
        entry.config(fg='light gray')


def btn_handle():
    print(entry.get())
    mlc = MLController()
    name = entry.get()
    modelname = "model_" + name
    vectorizername = "vector_" + name
    print(os.listdir(os.getcwd()))
    model, vectorizer = mlc.trainmodel_excel('Machine_Learning/Corrected_2_Updated_Preferred_titles.xlsx')
    # model, vectorizer = mlc.trainmodel_database('Machine_Learning/Corrected_2_Updated_Preferred_titles.xlsx')
    mlc.savemodel(model, vectorizer, modelname, vectorizername)


def btn_handle_trainfromdb():
    mlc = MLController()
    name = entry.get()
    modelname = "model_" + name
    vectorizername = "vector_" + name
    model, vectorizer = mlc.trainmodel_database()
    mlc.savemodel(model, vectorizer, modelname, vectorizername)


entry = tk.Entry(newwindow, fg='#c68be8', width=60, bg='#141017', borderwidth=0)
entry.insert(0, 'Modelname')
entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_focusout)

labelTitle.grid(row=0, column=0)
entry.grid(row=1, column=0, padx=30, pady=10)

button_confirm = tk.Button(newwindow, width=50, text="Train model again - from excel", command=btn_handle, fg='#c68be8', bg='#2f1e38', relief='flat', highlightcolor='#24182b', highlightbackground='#24182b', activebackground='#c68be8')
button_confirm.grid(row=2, column=0, columnspan=2)

button_confirm_db = tk.Button(newwindow, width=50, text="Train model again - from database", command=btn_handle_trainfromdb, fg='#c68be8', bg='#2f1e38', relief='flat', highlightcolor='#24182b', highlightbackground='#24182b', activebackground='#c68be8')
button_confirm_db.grid(row=3, column=0, columnspan=2, pady=10)

newwindow.mainloop()
