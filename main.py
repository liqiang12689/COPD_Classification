import utils

# dicom_path = "F:\\SZTU\\呼吸专科\\数据\\广医数据\\CTDATA\\E0001001V1\\E0001001V1FC03"
dicom_path = "F:\\Others\\E0001001V1FC03"
input_image = utils.get_input_image(dicom_path)

print(type(input_image))