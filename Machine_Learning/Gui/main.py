from Unparted_file import UnpartedFile
from Parted_file import PartedFile
from MenuHoster import MenuHoster

mode = "notconsole"

if mode == "console_UP":
    uf = UnpartedFile()
    uf.run_all()
elif mode == "console_P":
    pf = PartedFile()
    pf.run_all()
else:
    menu = MenuHoster()
    menu.hostmenu()
