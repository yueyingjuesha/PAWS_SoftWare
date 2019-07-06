import sys
from os import getcwd, system
from os.path import join
from run_makedata import main_predict, main_prep_qgis
import time


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QFileDialog, QMessageBox



class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.setWindowTitle(name)
        self.cwd = getcwd() 
        self.resize(300, 100) 

        self.chosen_model = None
        self.output = None
        self.chosen_file = None
        self.save_path = None
        self.has_result = False



        ## btn
        self.label1 = QLabel("Choose File:", self)
        self.btn_chooseFile = QPushButton("Choose File", self)  

        self.label2 = QLabel("Select Model:", self)
        self.btn_selectModel = QComboBox(self)  
        self.btn_selectModel.addItems(['Choose Model', 'XGBOOST','DECISION TREE','SVM'])

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
        self.chosen_file = QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./") 
        self.btn_chooseFile.setText(self.chosen_file)
        if self.chosen_model and self.chosen_file:
            self.btn_runModel.setEnabled(True)
        else:
            self.btn_runModel.setEnabled(False)
        return


    def slot_btn_selectModel(self, text):
        mapping = {'Choose Model': None, 'XGBOOST':'xgb','DECISION TREE':'dt','SVM':'svm'}
        self.chosen_model = mapping[text]
        if self.chosen_file and self.chosen_model:
            self.btn_runModel.setEnabled(True)
        else:
            self.btn_runModel.setEnabled(False)
        return


    def slot_btn_runModel(self):
        mapping = {'xgb':'XGBOOST','dt':'DECISION TREE','svm':'SVM'}
        QMessageBox.information(self, 'info1', 'Running {}, please wait'.format(mapping[self.chosen_model]))

        self.btn_runModel.setEnabled(False)
        self.btn_chooseFile.setEnabled(False)
        self.btn_selectModel.setEnabled(False)

        self.output = main_predict(self.chosen_file, self.chosen_model)

        self.btn_runModel.setEnabled(True)
        self.btn_chooseFile.setEnabled(True)
        self.btn_selectModel.setEnabled(True)
        if self.output[0] ==  False:
            QMessageBox.information(self, 'info3', 'No such a file in selected path: {}.csv'.format(self.output[1]))
        else:
            self.has_result = True
            if self.save_path:
                self.btn_exportResult.setEnabled(True)
            QMessageBox.information(self, 'info1', 'Running {} finished'.format(mapping[self.chosen_model]))
        return

    def slot_btn_exportResult(self):
        QMessageBox.information(self, 'info2', 'Results are saved in \n{}'.format(self.save_path))
        yea, mon, day, hou, minu, sec = list(time.localtime())[:6]
        name = '/PAWS%d_%02d_%02d_%02d_%02d_%02d.asc'%(yea, mon, day, hou, minu, sec)
        main_prep_qgis(self.output, self.save_path+name)
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
        reply = QMessageBox.question(self,
                                               'exit',
                                               "Do you want to exit?",
                                               QMessageBox.Yes | QMessageBox.No,
                                               QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('Demo V1.0')
    mainForm.show()
    sys.exit(app.exec_())
