import subprocess
import os


class QgisStandalone(object):
	"""
	Provide constructor with qgis install path, input .shp path, output shapefiles path, output .csv path
	The path format is provided as follows, you should initialize this object by providing your own path.
	Note that QgisStandalone.py and automate_data.py should be in the same folder, bash file is also generated in module folder
	"""
	def __init__(self, qgis_install_path="C:\\Program Files (x86)\\QGIS 2.18",
				 qgis_input_shp_path="C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS\\Quick_Employment\\QuickEmployment_Toy\\auto_input",
				 qgis_output_shapefile_path="C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS\\Quick_Employment\\QuickEmployment_Toy\\shapefiles",
				 qgis_output_csv_path="C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS\\Quick_Employment\\QuickEmployment_Toy\\csv_output"
				 ):

		super(QgisStandalone, self).__init__()
		self.qgis_install_path = qgis_install_path
		self.qgis_sub_install_path = self.qgis_install_path + "\\apps\\qgis-ltr"
		self.qgis_env_bat_path = self.qgis_install_path + "\\bin\\o4w_env.bat"

		self.qgis_input_shp_path = qgis_input_shp_path.replace("\\", "/")
		self.qgis_output_shapefile_path = qgis_output_shapefile_path.replace("\\", "/")
		self.qgis_output_csv_path = qgis_output_csv_path.replace("\\", "/")

		self.qgis_bash_path = os.path.dirname(os.path.realpath(__file__)) + "\\env_test.bat"
		self.qgis_automate_data_path = os.path.dirname(os.path.realpath(__file__)) + "\\automate_data.py"

	def run(self):
		self.make_dir()
		qgis_bash_script = self.generate_bash_script()
		self.check_path()
		session = subprocess.Popen(qgis_bash_script, stdout=subprocess.PIPE, shell=True)
		output, error = session.communicate()
		print('*' * 10 + 'output:' + '*' * 10)
		print(output)
		print('*' * 10 + 'error:' + '*' * 10)
		print(error)

	def make_dir(self):
		if not os.path.exists(self.qgis_output_shapefile_path):
			os.mkdir(self.qgis_output_shapefile_path)
		if not os.path.exists(self.qgis_output_csv_path):
			os.mkdir(self.qgis_output_csv_path)

	def check_path(self):
		if not os.path.exists(self.qgis_install_path):
			raise Exception('Qgis Install Path: \"' + self.qgis_install_path + '\" not found, make sure you have installed QGIS V2.18-ltr')

		if not os.path.exists(self.qgis_sub_install_path):
			raise Exception('Qgis Sub Install Path: \"' + self.qgis_sub_install_path + '\" not found, make sure you have installedrr QGIS V2.18-ltr')

		if not os.path.exists(self.qgis_env_bat_path):
			raise Exception('Qgis env_bat: \"' + self.qgis_env_bat_path + '\" not found')

		if not os.path.isfile(self.qgis_automate_data_path):
			raise Exception('automate_data.py file: \"' + self.qgis_automate_data_path + '\" not found')

		if not os.path.exists(self.qgis_input_shp_path):
			raise Exception('Qgis .shp input path: \"' + self.qgis_input_shp_path + '\" not found')

		if not os.path.exists(self.qgis_output_shapefile_path):
			raise Exception('Qgis shapefile output path \"' + self.qgis_output_shapefile_path + '\" not found')

		if not os.path.exists(self.qgis_output_csv_path):
			raise Exception('Qgis output csv path: \"' + self.qgis_output_csv_path + '\" not found')

		if not os.path.isfile(self.qgis_bash_path):
			raise Exception('Qgis bash script file: \"' + self.qgis_bash_path + '\" not found')

	def generate_bash_script(self):
		"""
		write QGIS bash script:
		1. setup qgis standalone envrionment
		2. setup qgis data input folder
		3. setup qgis data output folder
		"""
		buffer = []
		buffer.append("SET OSGEO4W_ROOT=" + self.qgis_install_path + "\r\n")
		buffer.append("SET QGISNAME=qgis-ltr\r\n")
		buffer.append("SET QGIS=%s\r\n" % self.qgis_sub_install_path)
		buffer.append("SET QGIS_PREFIX_PATH=%QGIS%\r\n")
		# buffer.append("echo %PATH%\r\n")
		buffer.append("CALL \"%s\"\r\n" % self.qgis_env_bat_path)
		buffer.append("SET PATH=%PATH%;%QGIS%\\bin\r\n")
		buffer.append("SET PYTHONPATH=%QGIS%\\python;%PYTHONPATH%\r\n")
		buffer.append("SET PYTHONPATH=" + self.qgis_sub_install_path + "\\python\\plugins;%PYTHONPATH%\r\n")
		buffer.append("SET PYTHONPATH=" + self.qgis_sub_install_path + "\\plugins;%PYTHONPATH%\r\n")
		buffer.append("SET PYTHONPATH=" + self.qgis_sub_install_path + "\\python;%PYTHONPATH%\r\n")
		# buffer.append("echo %PATH%\r\n")
		buffer.append("python \"%s\" \"%s/\" \"%s/\" \"%s/\"\r\n" % (self.qgis_automate_data_path, self.qgis_input_shp_path, self.qgis_output_shapefile_path, self.qgis_output_csv_path))

		with open(self.qgis_bash_path, "w+") as f:
			print("Saving bash script to " + self.qgis_bash_path)
			f.write(''.join(buffer))

		return self.qgis_bash_path

if __name__ == "__main__":
	qgis_standalone = QgisStandalone()
	qgis_standalone.run()