import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# --- 1. SETTING UP THE ENVIRONMENT ---

# Load the pre-trained digit recognition model
# Ensure 'digit_recognizer.h5' is in the same directory as this script.
try:
    model = load_model('digit_model.h5')
except Exception as e:
    print(f"Error loading model 'digit_recognizer.h5': {e}")
    print("Please ensure the model file is in the same directory as the script.")
    exit()

# --- 2. CORE IMAGE PROCESSING AND UTILITY FUNCTIONS ---

def preprocess_image(img):
    """Converts image to grayscale, blurs, and applies adaptive thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # Adaptive thresholding is great for uneven lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def find_grid_contour(processed_img):
    """Finds the largest contour that is likely the Sudoku grid."""
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check if the contour is roughly square-like and large enough
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    if len(approx) == 4 and cv2.contourArea(largest_contour) > 2000: # Increased minimum area
        return approx
    return None

def order_points(pts):
    """Orders the 4 points of the contour in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference
    return rect

def warp_perspective(image, contour):
    """Applies a perspective transform to get a top-down view of the grid."""
    ordered_pts = order_points(contour.reshape(4, 2))
    (tl, tr, br, bl) = ordered_pts
    
    # Define the width and height of the new image (a perfect square)
    width = 450 # 9 cells * 50 pixels
    height = 450
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
        
    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix

def extract_digits(warped_grid):
    """Extracts each digit from the 81 cells of the warped grid."""
    board = np.zeros((9, 9), dtype="int")
    cell_size = warped_grid.shape[0] // 9

    # Convert to grayscale and threshold for digit extraction
    gray_warped = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)
    thresh_warped = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    for i in range(9):
        for j in range(9):
            # Crop the cell from the grid
            cell = thresh_warped[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            # Add a border to avoid cutting off digits at the edge
            cell = cv2.copyMakeBorder(cell, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Find contours in the cell to locate the digit
            contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour, assuming it's the digit
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # If the digit is reasonably sized, process it
                if w > 15 and h > 15 and cv2.contourArea(c) > 100:
                    digit_roi = cell[y:y+h, x:x+w]
                    
                    # Prepare the digit for the model (resize to 28x28)
                    digit_img = cv2.resize(digit_roi, (28, 28))
                    digit_img = digit_img.astype("float") / 255.0
                    digit_img = np.expand_dims(digit_img, axis=-1)
                    digit_img = np.expand_dims(digit_img, axis=0)
                    
                    # Predict the digit
                    prediction = model.predict(digit_img, verbose=0)
                    board[i, j] = np.argmax(prediction)
    return board

# --- 3. SUDOKU SOLVER (BACKTRACKING ALGORITHM) ---

def find_empty(board):
    """Finds an empty cell (represented by 0) in the Sudoku board."""
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col
    return None

def is_valid(board, num, pos):
    """Checks if a number is valid in a given position."""
    # Check row
    for j in range(len(board[0])):
        if board[pos[0]][j] == num and pos[1] != j:
            return False
    # Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False
    # Check 3x3 box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False
    return True

def solve_sudoku(board):
    """Solves the Sudoku board using a backtracking algorithm."""
    find = find_empty(board)
    if not find:
        return True  # Puzzle is solved
    else:
        row, col = find

    for i in range(1, 10):
        if is_valid(board, i, (row, col)):
            board[row][col] = i
            if solve_sudoku(board):
                return True
            board[row][col] = 0 # Backtrack
    return False

# --- 4. DISPLAYING THE SOLUTION (WITH SAFETY CHECK) ---

def display_solution(frame, solved_board, original_board, transform_matrix):
    """Overlays the solved numbers onto the original camera frame."""
    warped_width = 450
    warped_height = 450
    cell_size = warped_width // 9
    
    # --- START OF THE FIX ---
    # Safety Check: A singular matrix cannot be inverted.
    # This happens when the detected grid corners are collinear (on a line).
    # We check the determinant; if it's close to zero, the matrix is singular.
    determinant = np.linalg.det(transform_matrix)
    if abs(determinant) < 1e-6: # Using a small threshold for floating point numbers
        # If the matrix is singular, we can't draw the solution.
        # Just return the original frame for this one instant.
        return frame
    # --- END OF THE FIX ---

    # Create a blank image to draw the solution numbers on
    solution_overlay = np.zeros((warped_height, warped_width, 3), dtype=np.uint8)

    for i in range(9):
        for j in range(9):
            # If the cell was originally empty, draw the new number
            if original_board[i][j] == 0:
                text = str(solved_board[i][j])
                text_x = j * cell_size + (cell_size // 4)
                text_y = i * cell_size + (cell_size // 4) * 3
                cv2.putText(solution_overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Inverse perspective transform to map the solution back to the original frame's perspective
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    unwarped_solution = cv2.warpPerspective(solution_overlay, inv_transform_matrix, (frame.shape[1], frame.shape[0]))
    
    # Combine the original frame with the solution overlay
    result_frame = cv2.addWeighted(frame, 1, unwarped_solution, 1, 0)
    return result_frame


# --- 5. MAIN APPLICATION LOOP ---

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Store the solved board to prevent re-solving every frame
    solved_board_cache = None
    original_board_cache = None
    matrix_cache = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # Process frame to find grid
        processed = preprocess_image(frame)
        grid_contour = find_grid_contour(processed)

        if grid_contour is not None:
            # If we find a grid, this is our best guess for the puzzle
            cv2.drawContours(display_frame, [grid_contour], -1, (0, 255, 0), 3)
            
            warped_grid, matrix = warp_perspective(frame, grid_contour)
            unsolved_board = extract_digits(warped_grid)

            # If this is a new puzzle, solve it and cache the results
            if original_board_cache is None or not np.array_equal(unsolved_board, original_board_cache):
                board_to_solve = unsolved_board.copy()
                if solve_sudoku(board_to_solve):
                    solved_board_cache = board_to_solve
                    original_board_cache = unsolved_board
                else:
                    # Could not solve this board, reset
                    solved_board_cache = None
                    original_board_cache = None

            # Cache the matrix for continuous display
            matrix_cache = matrix
        
        # If we have a solved puzzle in cache, keep displaying it
        if solved_board_cache is not None and matrix_cache is not None:
             display_frame = display_solution(frame, solved_board_cache, original_board_cache, matrix_cache)

        cv2.imshow("Sudoku AI Solver (Press 'r' to reset, 'q' to quit)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Press 'q' to quit
            break
        if key == ord('r'): # Press 'r' to reset and solve a new puzzle
            solved_board_cache = None
            original_board_cache = None
            matrix_cache = None
            print("Resetting solver. Show a new puzzle.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()