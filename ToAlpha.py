import cv2
import numpy as np

# 1. 이미지 불러오기 (흰 배경 포함)
img = cv2.imread('/Users/joseoglae/hansung/25-1/CV/pythonProject/tattoo_1.png')

# 2. 흰 배경 마스크 만들기 (배경이 거의 흰색이면 제거)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 밝은 배경 제거

# 3. BGR + 알파 채널로 병합
b, g, r = cv2.split(img)
rgba = cv2.merge((b, g, r, alpha))

# 4. 저장
cv2.imwrite('/Users/joseoglae/hansung/25-1/CV/pythonProject/tattoo_alpha_1.png', rgba)

print("배경 제거 완료: tattoo_alpha_2.png로 저장됨")
