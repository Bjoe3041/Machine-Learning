import Unparted_file
import MenuHoster
import os
mode = "notconsole"
# os.chdir(os.path.abspath(os.path.join("..", "..") + "\\"))
machine_learning_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(machine_learning_dir)
print("["+machine_learning_dir+"]")

# Machine-Learning
#   Machine_Learning
#       Controller
#       Gui
#           main.py
#       Model
#       Saves
#       Corrected_2


if mode == "console":
    uf = Unparted_file.UnpartedFile()
    uf.run_all()
else:
    menu = MenuHoster.MenuHoster()
    menu.hostmenu()
