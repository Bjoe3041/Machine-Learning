from Unparted_file import UnpartedFile
from MenuHoster import MenuHoster

mode = "notconsole"

if mode == "console":
    uf = UnpartedFile()
    uf.run_all()
else:
    menu = MenuHoster()
    menu.hostmenu()
