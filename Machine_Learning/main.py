
from Parted_file import PartedFile
from MenuHoster import MenuHoster
from MLController import MLController
pf = PartedFile()

#mlc = MLController()
#model, vectorizer = mlc.trainmodel('Machine_Learning/Corrected_2_Updated_Preferred_titles.xlsx')
#
# modelname = "model_test1_gaming"
# vectorizername = "vector_test1_gaming"
# #mlc.savemodel(model, vectorizer, modelname, vectorizername)
# loadedmodel, loadedvectorizer = mlc.loadmodel(modelname, vectorizername)
#
# print(mlc.evaluate(loadedmodel, loadedvectorizer, "role"))

menu = MenuHoster()
menu.hostmenu()

#pf.run_all()

#from Unparted_file import UnpartedFile
#upf = UnpartedFile()
#upf.run_all()

