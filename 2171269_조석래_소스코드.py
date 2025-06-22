import cv2  # 컴퓨터 비전 처리를 위한 OpenCV 라이브러리
import mediapipe as mp  # Google에서 제공하는 실시간 인체 추적 라이브러리 (Pose, Hands 등)
import numpy as np  # 고속 수치 계산을 위한 NumPy 라이브러리

# MediaPipe에서 사용할 기능들을 가져옴
mp_pose = mp.solutions.pose         # 전신 포즈 인식 (관절 33개 추적)
mp_hands = mp.solutions.hands       # 손가락 포인트 인식 (각 손에 대해 21개 관절)
mp_drawing = mp.solutions.drawing_utils  # 관절 결과 시각화를 위한 유틸 도구

# 포즈 인식 객체 생성: 사람의 신체 주요 관절 위치 추적
pose = mp_pose.Pose(
    min_detection_confidence=0.5,   # 관절 감지에 필요한 최소 신뢰도 (0.0 ~ 1.0)
    min_tracking_confidence=0.5)    # 추적 지속을 위한 최소 신뢰도

# 손 인식 객체 생성: 손가락 관절 위치를 추적
hands = mp_hands.Hands(
    max_num_hands=1,                # 한 손만 인식하도록 설정 (2로 하면 양손도 가능)
    min_detection_confidence=0.5,   # 손 감지 신뢰도 기준
    min_tracking_confidence=0.5)    # 손 추적 지속 신뢰도 기준

# 컴퓨터에 연결된 기본 웹캠(장치 번호 0)에서 영상 스트림 캡처 시작
cap = cv2.VideoCapture(0)

# 사용할 문신 이미지 파일 경로 리스트
tattoo_files = ["tattoo_alpha.png", "tattoo_alpha_1.png", "tattoo_alpha_2.png"]

# 현재 선택된 문신 이미지 인덱스 (초기값은 첫 번째 이미지)
current_tattoo_index = 0

# 선택된 문신 이미지 파일을 불러오기 (4채널: BGRA, 알파 포함)
tattoo = cv2.imread(tattoo_files[current_tattoo_index], cv2.IMREAD_UNCHANGED)

# 문신 이미지가 제대로 로딩되지 않았을 경우 오류 출력 및 프로그램 종료
if tattoo is None:
    print("문신 이미지를 불러올 수 없습니다.")
    exit()

# 초기 문신 부위 설정: 1=왼팔, 2=오른팔, 3=가슴
# 초기값 2는 오른팔을 의미
selected_position = 2


# 함수
# 사용자가 선택한 문신 이미지 인덱스를 바꿀 수 있도록 하는 함수
def change_tattoo(index):
    global tattoo, current_tattoo_index  # 전역 변수 사용 선언

    # 인덱스가 tattoo_files 범위 내에 있을 때만 변경
    if 0 <= index < len(tattoo_files):
        current_tattoo_index = index  # 현재 선택된 문신 인덱스 업데이트
        # 해당 인덱스의 문신 이미지 다시 로드 (BGRA - 알파 채널 포함)
        tattoo = cv2.imread(tattoo_files[current_tattoo_index], cv2.IMREAD_UNCHANGED)

# 선택 박스 정의
# 화면 상단 및 하단에 '신체 부위 선택' 및 '문신 종류 선택' 박스를 시각적으로 생성
def create_selection_boxes(frame):
    """프레임 상에 선택 가능한 UI 박스를 그리고, 각각의 위치 정보를 딕셔너리로 반환"""

    height, width = frame.shape[:2]  # 프레임의 높이와 너비 추출 (픽셀 단위)
    box_width = 100                  # 박스의 가로 길이 (픽셀)
    box_height = 60                  # 박스의 세로 길이 (픽셀)
    margin = 10                      # 화면 끝과 박스 사이의 여백
    vertical_gap = 80               # 위/아래 박스 사이의 수직 간격 (현재 사용 안 함)

    # 각 박스의 좌표 정의 (x1, y1, x2, y2) - 네 꼭짓점 중 왼쪽 위와 오른쪽 아래
    boxes = {
        1: (margin, margin, margin + box_width, margin + box_height),  # 왼팔 선택 박스 (좌측 상단)
        2: (width - margin - box_width, margin, width - margin, margin + box_height),  # 오른팔 박스 (우측 상단)
        3: (width // 2 - box_width // 2, margin, width // 2 + box_width // 2, margin + box_height)  # 가슴 박스 (가운데 상단)
    }

    # 화면 하단에 배치할 타투 종류 선택 박스들 좌표 계산
    y_offset = height - box_height - margin  # 하단 기준 y 좌표
    boxes[4] = (margin, y_offset, margin + box_width, y_offset + box_height)  # 타투1
    boxes[5] = (width - margin - box_width, y_offset, width - margin, y_offset + box_height)  # 타투2
    boxes[6] = (width // 2 - box_width // 2, y_offset, width // 2 + box_width // 2, y_offset + box_height)  # 타투3

    # 각 박스에 들어갈 텍스트 라벨 정의
    text_dict = {
        1: "왼팔",
        2: "오른팔",
        3: "가슴",
        4: "타투1",
        5: "타투2",
        6: "타투3"
    }

    # 각 박스를 화면 위에 그리기
    for pos, (x1, y1, x2, y2) in boxes.items():
        # 선택된 박스는 초록색, 선택되지 않은 박스는 흰색 테두리
        color = (0, 255, 0) if selected_position == pos else (255, 255, 255)

        # 사각형 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 텍스트 위치와 크기 설정
        text = text_dict.get(pos, "")
        cv2.putText(frame, text,
                    (x1 + 5, y1 + box_height // 2),  # 박스 내 중앙쯤 위치
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)  # 폰트 크기, 색, 굵기

    return boxes  # 각 박스의 좌표 정보를 포함한 딕셔너리 반환



# 손가락 끝 좌표와 박스들의 좌표를 비교하여, 어떤 박스를 터치했는지 반환하는 함수
def check_finger_touch(finger_coord, boxes):
    x, y = finger_coord  # 손가락 끝의 x, y 좌표를 분리

    # 모든 선택 박스들에 대해 반복
    for pos, (x1, y1, x2, y2) in boxes.items():
        # 손가락 좌표가 해당 박스 안에 있는 경우
        if x1 < x < x2 and y1 < y < y2:
            return pos  # 해당 박스의 위치 번호(pos)를 반환

    return None  # 어느 박스도 터치하지 않았을 경우 None 반환



# 사용자가 선택한 각도(angle)와 크기 배율(scale)에 따라 문신 이미지를 회전 및 확대/축소하는 함수
def rotate_and_scale_tattoo(image, angle, scale):
    h, w = image.shape[:2]  # 이미지의 세로(height), 가로(width) 크기 추출

    center = (w // 2, h // 2)  # 이미지의 중심 좌표 (회전 기준점)

    # 2D 회전 변환 행렬을 생성 (OpenCV 함수 사용)
    # angle: 시계 반대 방향 회전 (도 단위), scale: 크기 배율
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 회전 이후의 이미지 크기를 계산하기 위해 회전 행렬에서 cos, sin 성분 추출
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 회전 후 이미지가 차지할 새로운 가로, 세로 크기 계산
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 이동 보정: 회전 후 중심이 이미지 중앙에 오도록 행렬의 이동 값 수정
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 위에서 정의한 변환 행렬을 이용해 이미지를 실제로 회전 및 스케일 조절
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return rotated  # 변환된 이미지 반환



# 문신 이미지를 실제 영상 프레임 위에 덮어씌우는 함수 (알파값 고려, 선택 부위가 팔이면 마스킹도 적용)
def overlay_image(background, foreground, x, y, selected_position, landmarks):
    # 문신 시작 좌표가 프레임을 벗어나면 아무 작업도 하지 않고 원본 반환
    if x >= background.shape[1] or y >= background.shape[0]:
        return background

    h, w = foreground.shape[:2]  # 문신 이미지의 높이, 너비 추출

    # 만약 문신 이미지 전체가 화면 바깥쪽에 있을 경우, 합성할 필요 없음
    if y + h < 0 or x + w < 0:
        return background

    # 문신을 덮어씌울 좌표 범위 계산 (배경과 문신 이미지가 겹치는 실제 영역)
    y1 = max(0, y)
    y2 = min(background.shape[0], y + h)
    x1 = max(0, x)
    x2 = min(background.shape[1], x + w)

    # 문신 이미지 내에서 사용할 영역 좌표 계산 (배경과 맞춰 자르기)
    foreground_y1 = y1 - y
    foreground_y2 = y2 - y
    foreground_x1 = x1 - x
    foreground_x2 = x2 - x

    # 문신 이미지의 알파 채널(투명도)을 정규화하여 추출 (0.0 ~ 1.0)
    alpha = foreground[:, :, 3] / 255.0
    alpha = alpha[foreground_y1:foreground_y2, foreground_x1:foreground_x2]  # 사용할 영역만 잘라냄
    alpha = alpha[:, :, np.newaxis]  # 브로드캐스팅을 위한 차원 추가 (H x W x 1)

    # 만약 문신 부위가 팔이면 (왼팔 또는 오른팔), 배경 마스킹을 통해 문신이 팔에만 붙도록 제한
    if selected_position in [1, 2]:
        # 관절 좌표 (elbow, wrist)를 MediaPipe의 랜드마크로부터 추출
        if selected_position == 1:
            elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * background.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * background.shape[0]))
            wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * background.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * background.shape[0]))
        else:
            elbow = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * background.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * background.shape[0]))
            wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * background.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * background.shape[0]))

        # 팔 방향 벡터 계산 및 정규화
        arm_vector = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        arm_length = np.linalg.norm(arm_vector)
        arm_unit = arm_vector / arm_length  # 단위 벡터

        # 팔에 수직인 벡터 (문신이 양쪽으로 퍼질 방향)
        arm_normal = np.array([-arm_unit[1], arm_unit[0]])

        # 문신을 입힐 팔 너비 (팔 길이의 20%)
        forearm_width = arm_length * 0.2

        # 배경에서 문신을 합성할 영역에 대한 마스크 생성
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)

        # 마스크를 픽셀 단위로 계산 (팔 중심선으로부터 거리 기반 감쇠)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                px = x1 + j  # 전체 프레임 기준 x 좌표
                py = y1 + i  # 전체 프레임 기준 y 좌표
                rel_point = np.array([px - elbow[0], py - elbow[1]])  # 팔꿈치 기준 상대 좌표
                proj_length = np.dot(rel_point, arm_unit)  # 팔 방향 투영 길이

                if 0 <= proj_length <= arm_length:
                    dist_from_arm = abs(np.dot(rel_point, arm_normal))  # 팔에서 떨어진 거리
                    mask[i, j] = max(0, 1 - (dist_from_arm / (forearm_width / 2)))  # 1에서 거리에 비례해 감쇠

        mask = mask[:, :, np.newaxis]  # 브로드캐스팅을 위해 채널 추가
        alpha = alpha * mask  # 알파 채널에 마스크 적용 (팔 영역 이외는 투명)

    # 문신 이미지(foreground)와 영상 프레임(background)을 알파 블렌딩으로 합성
    for c in range(3):  # R, G, B 채널에 대해 반복
        background[y1:y2, x1:x2, c] = (
            alpha[:, :, 0] * foreground[foreground_y1:foreground_y2, foreground_x1:foreground_x2, c] +
            (1 - alpha[:, :, 0]) * background[y1:y2, x1:x2, c]
        )

    return background  # 최종 합성된 프레임 반환



def get_body_part_info(landmarks, selected_position):
    # 선택된 신체 부위에 따라 문신을 부착할 중심 좌표(center), 회전 각도(angle), 스케일 비율(scale)을 계산
    if selected_position == 1:
        # 어깨, 팔꿈치, 손목 좌표 추출 (0~1 정규화 → 화면 크기 곱해서 픽셀로 변환)
        shoulder = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
        elbow = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0]))
        wrist = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))

        # 팔뚝 (elbow-wrist) 길이 계산
        forearm_length = np.sqrt((wrist[0] - elbow[0]) ** 2 + (wrist[1] - elbow[1]) ** 2)

        # 상완 (shoulder-elbow) 길이 계산
        upperarm_length = np.sqrt((shoulder[0] - elbow[0]) ** 2 + (shoulder[1] - elbow[1]) ** 2)

        # 더 긴 쪽에 문신을 붙이기: 팔뚝이 길면 팔뚝, 아니면 상완
        if forearm_length > upperarm_length:
            center = ((elbow[0] + wrist[0]) // 2, (elbow[1] + wrist[1]) // 2)  # 중심 위치
            angle = np.degrees(np.arctan2(elbow[1] - wrist[1], elbow[0] - wrist[0]))  # 팔 방향 각도
            target_length = forearm_length
        else:
            center = ((shoulder[0] + elbow[0]) // 2, (shoulder[1] + elbow[1]) // 2)
            angle = np.degrees(np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0]))
            target_length = upperarm_length

        # 문신 크기 비율 계산
        tattoo_aspect_ratio = tattoo.shape[1] / tattoo.shape[0]  # width / height
        target_width = target_length * 0.4  # 문신 폭은 팔 길이의 40%
        target_height = target_width / tattoo_aspect_ratio       # 비율 유지
        scale = target_width / tattoo.shape[1]                   # 원본 대비 확대 비율

    elif selected_position == 2:
        shoulder = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))
        elbow = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]))
        wrist = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))

        forearm_length = np.sqrt((wrist[0] - elbow[0]) ** 2 + (wrist[1] - elbow[1]) ** 2)
        upperarm_length = np.sqrt((shoulder[0] - elbow[0]) ** 2 + (shoulder[1] - elbow[1]) ** 2)

        if forearm_length > upperarm_length:
            center = ((elbow[0] + wrist[0]) // 2, (elbow[1] + wrist[1]) // 2)
            angle = np.degrees(np.arctan2(elbow[1] - wrist[1], elbow[0] - wrist[0]))
            target_length = forearm_length
        else:
            center = ((shoulder[0] + elbow[0]) // 2, (shoulder[1] + elbow[1]) // 2)
            angle = np.degrees(np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0]))
            target_length = upperarm_length

        tattoo_aspect_ratio = tattoo.shape[1] / tattoo.shape[0]
        target_width = target_length * 0.4
        target_height = target_width / tattoo_aspect_ratio
        scale = target_width / tattoo.shape[1]

    else:
        left_shoulder = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
        right_shoulder = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))

        chest_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2)

        # 가슴은 위에서 보기에 맞게 180도 회전 (거꾸로 보이는 걸 방지)
        angle = np.degrees(np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0])) + 180

        shoulder_width = np.sqrt(
            (right_shoulder[0] - left_shoulder[0]) ** 2 +
            (right_shoulder[1] - left_shoulder[1]) ** 2)

        tattoo_aspect_ratio = tattoo.shape[1] / tattoo.shape[0]
        target_width = shoulder_width * 0.4  # 어깨 폭의 40%를 문신 폭으로 설정
        target_height = target_width / tattoo_aspect_ratio
        scale = target_width / tattoo.shape[1]
        center = chest_center

    return center, angle, scale  # 최종 계산된 위치, 각도, 스케일 반환



# ----------------------------
# 메인 루프: 프레임을 반복적으로 읽고 처리
# ----------------------------
while cap.isOpened():  # 웹캠이 정상적으로 열려 있으면 계속 실행
    ret, frame = cap.read()  # 프레임 캡처 시도 (ret: 성공 여부, frame: 이미지)

    if not ret:  # 프레임을 읽지 못했을 경우 (예: 카메라 연결 해제)
        print("카메라를 찾을 수 없습니다.")
        break  # 루프 종료

    # 프레임을 좌우 반전하여 사용자가 거울 보듯 보이도록 설정 (편의성 향상)
    frame = cv2.flip(frame, 1)

    # OpenCV는 BGR 채널, MediaPipe는 RGB 채널을 요구 → 변환 필요
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 포즈 탐지: 전체 몸에서 관절 위치 추출
    pose_results = pose.process(frame_rgb)

    # 손 탐지: 손가락 위치 (특히 검지 끝)를 추출
    hand_results = hands.process(frame_rgb)

    # 상단/하단 UI 박스들을 화면에 생성하고, 각 박스의 좌표 반환
    selection_boxes = create_selection_boxes(frame)

    # 손가락 터치 감지: 검지가 어떤 박스를 터치했는지 확인
    if hand_results.multi_hand_landmarks:  # 손이 감지되었을 경우
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # 검지 끝

            # 검지 위치를 프레임 좌표계로 변환 (정규화 → 픽셀 단위)
            finger_x = int(index_finger.x * frame.shape[1])
            finger_y = int(index_finger.y * frame.shape[0])

            # 손가락 끝 위치를 시각적으로 확인하기 위해 원(circle) 표시
            cv2.circle(frame, (finger_x, finger_y), 8, (255, 0, 0), -1)

            # 손가락이 어떤 박스를 터치했는지 확인
            touched_position = check_finger_touch((finger_x, finger_y), selection_boxes)

            if touched_position is not None:
                if touched_position in [1, 2, 3]:
                    # 1~3번 박스는 신체 부위 선택 (왼팔/오른팔/가슴)
                    selected_position = touched_position
                elif touched_position in [4, 5, 6]:
                    # 4~6번 박스는 문신 종류 선택
                    change_tattoo(touched_position - 4)  # tattoo_files[0~2]에 매핑

    # 문신 합성 단계: 포즈가 감지된 경우에만 실행
    if pose_results.pose_landmarks:
        # 관절 정보 리스트
        landmarks = pose_results.pose_landmarks.landmark

        # 현재 선택된 신체 부위에 따라 문신 위치/각도/크기 계산
        center, angle, scale = get_body_part_info(landmarks, selected_position)

        # 문신 이미지 회전 및 스케일 조정
        transformed_tattoo = rotate_and_scale_tattoo(tattoo, angle, scale)

        # 회전된 문신 이미지의 좌상단 좌표 계산 (중앙 기준으로 좌표 조정)
        tattoo_x = center[0] - transformed_tattoo.shape[1] // 2
        tattoo_y = center[1] - transformed_tattoo.shape[0] // 2

        # 문신 이미지를 실제 프레임에 합성 (마스킹 포함)
        frame = overlay_image(frame, transformed_tattoo, tattoo_x, tattoo_y, selected_position, landmarks)

    # 최종 프레임 출력 (문신 포함)
    cv2.imshow('Tattoo Overlay', frame)  # 'Tattoo Overlay' 창에 결과 표시

    # ESC 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 리소스 정리 및 종료 처리
cap.release()              # 웹캠 장치 해제
cv2.destroyAllWindows()    # OpenCV 창 모두 닫기
pose.close()               # MediaPipe 포즈 객체 종료
hands.close()              # MediaPipe 손 객체 종료
