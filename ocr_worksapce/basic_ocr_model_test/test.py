import easyocr

reader = easyocr.Reader(['ko']) # need to run only once to load model into memory
result = reader.readtext('./hello2.jpg')

print(result)
