from QgisIntegration.QgisStandalone import QgisStandalone

qgis = QgisStandalone(qgis_install_path="C:\\Users\\MaxWillx\\CMU course\\Paws Project\\QGIS 2.18",
				 qgis_input_shp_path="C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS_SoftWare\\QgisIntegration\\auto_input",
				 qgis_output_shapefile_path="C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS_SoftWare\\QgisIntegration\\shapefiles123",
				 qgis_output_csv_path="C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS_SoftWare\\QgisIntegration\\csvfiles123"
				 )
qgis.run()