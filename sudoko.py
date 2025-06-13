from tensorflow.keras.models import load_model # type: ignore
import cv2
import numpy as np
import operator
import sudoku_solver as sol 

try:
    classifier = load_model("digit_model.h5")
except Exception as e:
    print(f"Error: Could not load model 'digit_model.h5'. {e}")
    print("Please ensure the model file is in the correct directory.")
    exit()

marge = 4
case = 28 + 2 * marge
taille_grille = 9 * case

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()


fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1080, 620))

is_solved = False
result = None
original_grille_txt = None

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Image Processing ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_grille = None
    maxArea = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
            if area > maxArea and len(polygone) == 4:
                contour_grille = polygone
                maxArea = area

    # --- Grid Found ---
    if contour_grille is not None:
        # CHANGED COLOR: The grid border is now dark blue (255, 0, 0)
        cv2.drawContours(frame, [contour_grille], 0, (255, 0, 0), 2)
        
        points = np.vstack(contour_grille).squeeze() # type: ignore
        if len(points) == 4:
            points = sorted(points, key=operator.itemgetter(1))
            if points[0][0] < points[1][0]:
                pts1 = np.float32([points[0], points[1], points[3], points[2]]) if points[3][0] < points[2][0] else np.float32([points[0], points[1], points[2], points[3]])  # type: ignore
            else:
                pts1 = np.float32([points[1], points[0], points[3], points[2]]) if points[3][0] < points[2][0] else np.float32([points[1], points[0], points[2], points[3]])  # type: ignore
            
            pts2 = np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [taille_grille, taille_grille]])  # type: ignore
            M = cv2.getPerspectiveTransform(pts1, pts2)   # type: ignore
            grille_warped = cv2.warpPerspective(frame, M, (taille_grille, taille_grille))
            grille_processed = cv2.cvtColor(grille_warped, cv2.COLOR_BGR2GRAY)
            grille_processed = cv2.adaptiveThreshold(grille_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

            # --- Solve the puzzle if not already solved ---
            if not is_solved:
                grille_txt_list = []
                for y in range(9):
                    ligne = ""
                    for x in range(9):
                        y2min, y2max = y * case + marge, (y + 1) * case - marge
                        x2min, x2max = x * case + marge, (x + 1) * case - marge
                        img_cell = grille_processed[y2min:y2max, x2min:x2max]
                        img_reshaped = img_cell.reshape(1, 28, 28, 1)
                        if img_reshaped.sum() > 10000:
                            prediction_probs = classifier.predict(img_reshaped, verbose=0)
                            prediction = np.argmax(prediction_probs)
                            ligne += str(prediction)
                        else:
                            ligne += "0"
                    grille_txt_list.append(ligne)
                
                print("Board detected:", grille_txt_list)
                original_grille_txt = grille_txt_list[:]
                result = sol.sudoku(grille_txt_list)
                
                if result is not None:
                    print("Solution found!")
                    is_solved = True
            
            # --- Display the solution ---
            if is_solved and result is not None:
                # 1. Create a blank image to draw the solution numbers on
                fond = np.zeros((taille_grille, taille_grille, 3), dtype=np.uint8)
                for y in range(len(result)):
                    for x in range(len(result[y])):
                        if original_grille_txt[y][x] == "0": # Only draw new numbers  # type: ignore
                            text = str(result[y][x])
                            pos_x = x * case + marge + 3
                            pos_y = (y + 1) * case - marge - 3
                            # CHANGED COLOR: The solution text is now dark blue (255, 0, 0)
                            cv2.putText(fond, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # 2. Un-warp the solution to match the original grid's perspective
                M_inv = cv2.getPerspectiveTransform(pts2, pts1)  # type: ignore
                h, w, _ = frame.shape
                solution_overlay = cv2.warpPerspective(fond, M_inv, (w, h))
                
                # 3. Combine the frame and the overlay
                frame = cv2.addWeighted(frame, 1, solution_overlay, 1, 0)

    # --- No Grid Found ---
    else:
        is_solved = False
        result = None
        original_grille_txt = None

    # --- Show and Save Frame ---
    display_frame = cv2.resize(frame, (1080, 620))
    cv2.imshow("Sudoku Solver (Press 'q' to quit)", display_frame)
    out.write(display_frame)

    # --- Quit Condition ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- Cleanup ---
out.release()
cap.release()
cv2.destroyAllWindows()