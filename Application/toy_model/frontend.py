import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from code import toy_runner
import numpy as np

class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.setWindowTitle(name)
        self.cwd = os.getcwd() 
        self.resize(300, 100) 

        self.chosen_model = None
        self.has_result = None
        self.chosen_file = None
        self.save_path = None
        self.runner = None



        ## btn
        self.label1 = QLabel("Choose File:", self)
        self.btn_chooseFile = QPushButton("Choose File", self)  

        self.label2 = QLabel("Select Model:", self)
        self.btn_selectModel = QComboBox(self)  
        self.btn_selectModel.addItems(['model1','model2','model3'])

        self.btn_runModel = QPushButton("Run Model", self) 

        self.label3 = QLabel("Choose Save Path:", self)
        self.btn_chooseDir = QPushButton("Choose Save Path", self)  

        self.btn_exportResult = QPushButton("Export Result", self)  

        self.btn_runModel.setEnabled(False)
        self.btn_exportResult.setEnabled(False)





        layout = QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.btn_chooseFile)
        layout.addWidget(self.label2)
        layout.addWidget(self.btn_selectModel)
        layout.addWidget(self.btn_runModel)
        layout.addWidget(self.label3)
        layout.addWidget(self.btn_chooseDir)
        layout.addWidget(self.btn_exportResult)
        self.setLayout(layout)


        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)
        self.btn_selectModel.activated[str].connect(self.slot_btn_selectModel)
        self.btn_runModel.clicked.connect(self.slot_btn_runModel)
        self.btn_chooseDir.clicked.connect(self.slot_btn_chooseDir)
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
        self.btn_chooseFile.setText(self.chosen_file)
        if self.chosen_model:
            self.btn_runModel.setEnabled(True)
        return


    def slot_btn_selectModel(self, text):
        self.chosen_model = text
        if self.chosen_file:
            self.btn_runModel.setEnabled(True)
        return


    def slot_btn_runModel(self):
        QMessageBox.information(self, 'info1', 'Running {}, please wait'.format(self.chosen_model))
        for i in range(10000):
            a = np.random.rand(100000)
        model_type = int(self.chosen_model[-1])
        self.runner = toy_runner(model_type, self.chosen_file)
        self.runner.run()
        QMessageBox.information(self, 'info0', 'Running finished!'.format(self.chosen_model))
        self.has_result = self.runner.check_result()
        if self.save_path and self.has_result:
            self.btn_exportResult.setEnabled(True)
        return

    def slot_btn_exportResult(self):
        QMessageBox.information(self, 'info2', 'Results are saved in \n{}'.format(self.save_path))    
        self.runner.save_result(self.save_path)
        return

    def slot_btn_chooseDir(self):
        dir_choose = QFileDialog.getExistingDirectory(self,  
                                    "Choose Path",  
                                    self.cwd) 

        self.save_path = dir_choose
        self.btn_chooseDir.setText(dir_choose)
        if self.save_path and self.has_result:
            self.btn_exportResult.setEnabled(True)
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
