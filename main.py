import utils

dicom_path = "/data/zengnanrong/CTDATA/E0001001V1"
input_image = utils.get_input_image(dicom_path)

print(type(input_image))