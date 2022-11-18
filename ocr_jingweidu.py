import easyocr
import cv2

img_path = 'D:\\Data\\jiujiang\\DJI_202208020740_005\\DJI_20220802074324_0001_W.JPG'
img = cv2.imread(img_path)
img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_recon = img[:400, :400]

# img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('test.jpg', img)


reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext(img_recon)
result_zuobiao = result[-1]
result_zuobiao_ = str(result_zuobiao[1]).strip().split('N')
latitude_co = result_zuobiao_[0][:9]
longitude_co = result_zuobiao_[1].strip()[:9]
print(latitude_co, longitude_co)
# print(result_zuobiao_)
# print(result)