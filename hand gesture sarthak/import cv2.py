import cv2
import mediapipe as mp

# Function to classify hand gestures
def classify_gesture(lm_list):
    # Calculate the distances between finger joints
    thumb_dist = lm_list[4][1] - lm_list[3][1]
    index_dist = lm_list[8][1] - lm_list[6][1]
    middle_dist = lm_list[12][2] - lm_list[10][2]
    ring_dist = lm_list[16][2] - lm_list[14][2]
    little_dist = lm_list[20][2] - lm_list[18][2]

    # Classify gestures based on finger positions
    if thumb_dist < 0 and index_dist < 0 and middle_dist < 0 and ring_dist < 0 and little_dist < 0:
        gesture = "Fist"
    elif thumb_dist > 0 and index_dist < 0 and middle_dist < 0 and ring_dist < 0 and little_dist < 0:
        gesture = "Thumb Up"
    elif thumb_dist < 0 and index_dist > 0 and middle_dist > 0 and ring_dist > 0 and little_dist > 0:
        gesture = "Full Open Hand"
    else:
        gesture = "Unknown"

    return gesture

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand landmark positions
            lm_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # Classify gesture
            gesture = classify_gesture(lm_list)

            # Display the gesture
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
