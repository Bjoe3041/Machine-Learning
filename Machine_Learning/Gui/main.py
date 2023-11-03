import MenuHoster
import os
import subprocess
from Unparted_file import UnpartedFile
from Parted_file import PartedFile
from MenuHoster import MenuHoster

mode = "notconsole"
# os.chdir(os.path.abspath(os.path.join("..", "..") + "\\"))
machine_learning_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(machine_learning_dir)
print("[" + machine_learning_dir + "]")


def topercent(val1, val2):
    valsum = val1 + val2
    x = 1 / valsum
    return val1 * x, val2 * x


p = topercent(0.7, 0.3)
print(p)


# Machine-Learning
#   Machine_Learning
#       Controller
#       Gui
#           main.py
#       Model
#       Saves
#       Corrected_2

def install_dependencies():
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully. // Dependencies checked successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. You're fucked...")


install_dependencies()

if mode == "console_UP":
    uf = UnpartedFile()
    uf.run_all()
elif mode == "console_P":
    pf = PartedFile()
    pf.run_all()
else:
    menu = MenuHoster()
    menu.hostmenu()
