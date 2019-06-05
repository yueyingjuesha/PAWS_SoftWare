import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from code import toy_runner

class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.setWindowTitle(name)
        self.cwd = os.getcwd() 
        self.resize(300, 100) 

        self.chosen_model = None
        self.has_result = None
        self.chosen_file = None
        self.runner = None



        ## btn
        self.btn_chooseFile = QPushButton("Choose File", self)  

        self.label = QLabel("Select Model:", self)
        self.btn_selectModel = QComboBox(self)  
        self.btn_selectModel.addItems(['model1','model2','model3'])

        self.btn_runModel = QPushButton("run Model", self) 

        self.btn_exportResult = QPushButton("Export Result", self)  





        layout = QVBoxLayout()
        layout.addWidget(self.btn_chooseFile)
        layout.addWidget(self.label)
        layout.addWidget(self.btn_selectModel)
        layout.addWidget(self.btn_runModel)
        layout.addWidget(self.btn_exportResult)
        self.setLayout(layout)


        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)
        self.btn_selectModel.activated[str].connect(self.slot_btn_selectModel)
        self.btn_runModel.clicked.connect(self.slot_btn_runModel)
        self.btn_exportResult.clicked.connect(self.slot_btn_exportResult)





    def slot_btn_chooseFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                    "Choose File",  
                                    self.cwd,
                                    "All Files (*);;Text Files (*.txt)")  

        if fileName_choose == "":
            print("\nCancel")
            return
        self.chosen_file = fileName_choose
        print("\nThe file you uploaded:")
        print(fileName_choose)


    def slot_btn_selectModel(self, text):
        self.chosen_model = text
        print("\nThe model you chose:")
        print(text)


    def slot_btn_runModel(self):
        if not self.chosen_file:
            QMessageBox.warning(self,'warm0', 'Please choose a file first')
            return
        if not self.chosen_model:
            QMessageBox.warning(self,'warm1', 'Please choose a model first')
            return
        else:
            QMessageBox.information(self, 'info1', 'Running {}, please wait'.format(self.chosen_model))
            model_type = int(self.chosen_model[-1])
            self.runner = toy_runner(model_type, self.chosen_file)
            self.runner.run()
            self.has_result = self.runner.check_result()
            return

    def slot_btn_exportResult(self):
        if not self.has_result:
            QMessageBox.warning(self,'warm2', 'Please run a model first')
            return
        else:
            QMessageBox.information(self, 'info2', 'Results are saved in \n{}'.format(self.cwd))    
            self.runner.save_result(self.cwd)
            return




    def closeEvent(self, event):  
        reply = QtWidgets.QMessageBox.question(self,
                                               'exit',
                                               "Do you want to exit?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('Demo V1.0')
    mainForm.show()
    sys.exit(app.exec_())
