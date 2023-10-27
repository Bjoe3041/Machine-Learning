from Unparted_file import UnpartedFile
from MenuHoster import MenuHoster

mode = "console"

if mode == "notconsole":
    uf = UnpartedFile()
    uf.run_all()
else:
    menu = MenuHoster()
    menu.hostmenu()

