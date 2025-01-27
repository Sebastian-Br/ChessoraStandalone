import cv2
import numpy as np
import pyautogui
import chess
import keyboard
import pygetwindow as gw
import win32api
import win32con
import win32gui
import time
import math
import chess.engine
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import os
import mss
import statistics
import sys
import re
import atexit
import ctypes
from ctypes import wintypes
import tkinter as tk
import threading
import subprocess
from fastai.vision.all import *

class HiddenWindowEngine(chess.engine.SimpleEngine):
    @classmethod
    def popen_uci(cls, command, setpgrp=False, **kwargs):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        # Update kwargs to include startupinfo
        kwargs['startupinfo'] = startupinfo

        return super().popen_uci(command, setpgrp=setpgrp, **kwargs)

stockfish_path = "stockfish16.1/stockfish-windows-x86-64-avx2.exe"
engine = HiddenWindowEngine.popen_uci(stockfish_path)

def get_label(x):
    # Define your get_label function here
    pass

exported_model_path = Path('m.pkl')
learn = load_learner(exported_model_path)

def PredictLabel_From50By50GrayScaleSquare(square):
    square = cv2.resize(square, (50, 50), interpolation=cv2.INTER_CUBIC)
    pred_class, _, outputs = learn.predict(square)
    max_certainty = outputs.max().item()
    if max_certainty < 0.98:
        pred_class = None
    #window_title = f"Class: {pred_class}, Certainty: {max_certainty:.2f}"
    #cv2.imshow(window_title, square)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return pred_class

previousFEN = ""
previousMoves = []

UserColor = None

engine_multi_pv = 0
engine_analysis_time_per_variation_seconds = 0.019
#draw vs chess.com lvl 25 as white: 0.0035 (1 thread) (10k nodes on test position)
engine_analysis_time = None

def configureEngine():
    engine.configure({"Threads": 1})

atexit.register(engine.quit)

mapping = {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
        '1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0
    }
    
mapping_black = {
        'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e': 3, 'f': 2, 'g': 1, 'h': 0,
        '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7
    }

def capture_screenshot(x, y, crop_x_left, crop_y_top, crop_width, crop_height):
    try:
        with mss.mss() as sct:
            monitor = {
                "left": x + crop_x_left,
                "top": y + crop_y_top,
                "width": crop_width,
                "height": crop_height
            }
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None

crop_left = 0.8
crop_right = 0.8
crop_top = 0.8
crop_bottom = 0.8

crop_x_left = 0
crop_x_right = 0
crop_y_top = 0
crop_y_bottom = 0
crop_width = 0
crop_height = 0

def set_screen_size():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    global crop_left, crop_right, crop_top, crop_bottom
    global crop_x_left, crop_x_right, crop_y_top, crop_y_bottom, crop_width, crop_height
    crop_x_left = int(screen_width * crop_left)
    crop_x_right = int(screen_width * crop_right)
    crop_y_top = int(screen_height * crop_top)
    crop_y_bottom = int(screen_height * crop_bottom)
    crop_width = screen_width - crop_x_left - crop_x_right
    crop_height = screen_height - crop_y_top - crop_y_bottom

def capture_chessboard():
    global crop_x_left, crop_x_right, crop_y_top, crop_y_bottom, crop_width, crop_height
    screenshot = capture_screenshot(0, 0, crop_x_left, crop_y_top, crop_width, crop_height)
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray_screenshot, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
    kernel = np.ones((3, 3), np.uint8)
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT , kernel)
    #closed_thresh = cv2.morphologyEx(closed_thresh, cv2.MORPH_CLOSE , kernel)

    contours, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finds only external contours. if the chessboard is itself in another contour you need to change this parameter.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    chessboard_contour = None
    #i = 0
    for contour in contours:
        #i += 1
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(contour)
        #print(f"Contour {i} w,h: {w},{h}")
        if len(approx) == 4:
            #x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.99 < aspect_ratio  < 1.01 and min(w, h) >= 320:  # Ensuring the contour is approximately square
                chessboard_contour = approx
                break
    if chessboard_contour is None:
        raise ValueError("Chessboard contour not found")
    x_rel, y_rel, w, h = cv2.boundingRect(chessboard_contour)
    x_abs = crop_x_left + x_rel
    y_abs = crop_y_top + y_rel
    chessboard = gray_screenshot[y_rel:y_rel+h, x_rel:x_rel+w]
    
    #cv2.imshow('Cropped Image', screenshot)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return chessboard, x_abs, y_abs, w, h

def create_background_color_array():
    # An 8x8 array representing the colors of the chessboard
    background_colors = [[(i + j) % 2 == 0 for j in range(8)] for i in range(8)]
    
    return background_colors

background_colors = create_background_color_array()

imageToPieceDictionary = {} #key: image bytes. value: piece, e.g. wR

def recognize_pieces(gray_chessboard, square_size):
    #gray_chessboard = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
    piece_positions = {}
    half_square_size = square_size // 2

    for row in range(8):
        for col in range(8):
            # Extract the square region
            y = row * square_size
            x = col * square_size
            gray_square = gray_chessboard[y:y + square_size, x:x + square_size]
            upper_left_quadrant = gray_square[:half_square_size, :half_square_size]

            # Define the upper-right half of the upper-left quadrant
            half_width = upper_left_quadrant.shape[1] // 2
            cropped_gray_square = upper_left_quadrant[:, half_width:]
            cropped_gray_square_bytes = cropped_gray_square.tobytes()
            if cropped_gray_square_bytes in imageToPieceDictionary:
                encodedPiece = imageToPieceDictionary[cropped_gray_square_bytes]
                if encodedPiece == "":
                    continue
                piece_positions[(row, col)] = encodedPiece
            else:
                #is_light_square = background_colors[row][col]
                piece = PredictLabel_From50By50GrayScaleSquare(gray_square)
                if piece is None:
                    imageToPieceDictionary[cropped_gray_square_bytes] = ""  # No match found
                else:
                    imageToPieceDictionary[cropped_gray_square_bytes] = piece
                    piece_positions[(row, col)] = piece

    #print("Piece Positions:\n")
    #print(piece_positions)
    #print("\n")
    return piece_positions


def generate_fen(piece_positions, CurrentIterationUserColor):
    board = chess.Board.empty()
    if CurrentIterationUserColor == chess.BLACK:
        for (row, col), piece in piece_positions.items():
            square = chess.square(7 - col, row)
            board.set_piece_at(square, chess.Piece.from_symbol(piece))
    else:
        for (row, col), piece in piece_positions.items():
            square = chess.square(col, 7 - row)
            board.set_piece_at(square, chess.Piece.from_symbol(piece))

    return board.fen()


def on_shift_w():
    global UserColor
    global UserColorChanged
    UserColor = chess.WHITE
    UserColorChanged = True
    print("SHIFT+W pressed. Setting user color to WHITE")

def on_shift_b():
    global UserColor
    global UserColorChanged
    UserColor = chess.BLACK
    UserColorChanged = True
    print("SHIFT+B pressed. Setting user color to BLACK")

user32 = ctypes.WinDLL('user32', use_last_error=True)

VK_SHIFT = 0x10
VK_W = 0x57
VK_B = 0x42

def is_key_pressed(vk_code):
    return user32.GetAsyncKeyState(vk_code) & 0x8000

def wait_for_picking_color():
    global UserColor
    print("Press 'SHIFT+W' to start the script as WHITE or 'SHIFT+B' to start as BLACK...")
    while UserColor is None:
        if is_key_pressed(VK_SHIFT):
            if is_key_pressed(VK_W):
                on_shift_w()
            if is_key_pressed(VK_B):
                on_shift_b()
        time.sleep(0.075)

def recheck_user_color__thread():
    while True:
        if is_key_pressed(VK_SHIFT):
            if is_key_pressed(VK_W):
                on_shift_w()
                time.sleep(0.08)
            if is_key_pressed(VK_B):
                on_shift_b()
                time.sleep(0.08)
        else:
            time.sleep(1)

def square_center_position(square_notation, x_abs, y_abs, square_size, CurrentIterationUserColor):
    global mapping
    global mapping_black
    
    if CurrentIterationUserColor == chess.WHITE:
        col_index, row_index = mapping[square_notation[0]], mapping[square_notation[1]]
    else:
        col_index, row_index = mapping_black[square_notation[0]], mapping_black[square_notation[1]]

    center_x = x_abs + (col_index + 0.5) * square_size
    center_y = y_abs + (row_index + 0.5) * square_size
    return (center_x, center_y)

draw_moves_thread = None

draw_moves_PARAM_moves = None
draw_moves_PARAM_x_abs = None
draw_moves_PARAM_y_abs = None
draw_moves_PARAM_square_size = None

def draw_moves_reuse_hwnd():
    arrows = []
    global CurrentIterationUserColor
    global draw_moves_PARAM_moves
    global draw_moves_PARAM_x_abs
    global draw_moves_PARAM_y_abs
    global draw_moves_PARAM_square_size
    
    for move in draw_moves_PARAM_moves:
        squareFrom = move['squares'][:2]  # Extract the first two characters
        squareTo = move['squares'][2:]    # Extract the last two characters
        squareFromPosition = square_center_position(squareFrom, draw_moves_PARAM_x_abs, draw_moves_PARAM_y_abs, draw_moves_PARAM_square_size, CurrentIterationUserColor)
        squareToPosition = square_center_position(squareTo, draw_moves_PARAM_x_abs, draw_moves_PARAM_y_abs, draw_moves_PARAM_square_size, CurrentIterationUserColor)
        arrow = (squareFromPosition[0], squareFromPosition[1], squareToPosition[0], squareToPosition[1], move['centipawnloss'], move['score'])
        arrows += [arrow]
    update_overlay_globally(arrows)
    show_overlay(draw_duration)

draw_duration = 1

def draw_arrow_globally(hdc, xstart, ystart, xend, yend):
        xstart = int(xstart)
        ystart = int(ystart)
        xend = int(xend)
        yend = int(yend)
        win32gui.MoveToEx(hdc, xstart, ystart)
        win32gui.LineTo(hdc, xend, yend)

        angle = math.atan2(yend - ystart, xend - xstart)

        arrow_length = 15
        arrow_angle = math.pi / 6  # 30 degrees

        x1 = int(xend - arrow_length * math.cos(angle - arrow_angle))
        y1 = int(yend - arrow_length * math.sin(angle - arrow_angle))
        x2 = int(xend - arrow_length * math.cos(angle + arrow_angle))
        y2 = int(yend - arrow_length * math.sin(angle + arrow_angle))

        win32gui.MoveToEx(hdc, xend, yend)
        win32gui.LineTo(hdc, x1, y1)
        win32gui.MoveToEx(hdc, xend, yend)
        win32gui.LineTo(hdc, x2, y2)

overlay_hwnd = None
current_arrows = []

def create_overlay_window_globally():
    global overlay_hwnd
    global current_arrows

    def wnd_proc(hwnd, msg, wParam, lParam):
        if msg == win32con.WM_PAINT:
            hdc, paint_struct = win32gui.BeginPaint(hwnd)
            
            # Clear the window with transparency
            rect = win32gui.GetClientRect(hwnd)
            brush = win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0))  # Black color (color key will make it transparent)
            win32gui.FillRect(hdc, rect, brush)
            win32gui.DeleteObject(brush)
            for xstart, ystart, xend, yend, centipawnloss, score in current_arrows:
                # Calculate color based on centipawnloss
                if centipawnloss <= 0:
                    green_value = 255
                    red_value = 0
                elif centipawnloss <= 50:
                    green_value = max(0, 255 - 5 * centipawnloss)
                    red_value = max(0, 5 * centipawnloss - 255)
                else:
                    green_value = 0
                    red_value = min(255, 5 * (centipawnloss - 50))

                red_value = int(red_value)
                green_value = int(green_value)
                pen_color = win32api.RGB(red_value, green_value, 0)
                pen = win32gui.CreatePen(win32con.PS_SOLID, 2, pen_color)
                win32gui.SelectObject(hdc, pen)

                draw_arrow_globally(hdc, xstart, ystart, xend, yend)

                text_color = win32api.RGB(0, 0, 255) #blue
                win32gui.SetTextColor(hdc, text_color)
                
                win32gui.SetBkMode(hdc, win32con.TRANSPARENT) #transparent background mode

                # Set a small font
                logfont = win32gui.LOGFONT()
                logfont.lfHeight = -14
                font = win32gui.CreateFontIndirect(logfont)
                win32gui.SelectObject(hdc, font)

                # Draw the score near the end of the arrow
                text = f"{score:+.2f}"
                text_x = int(xend + 6)
                text_y = int(yend + 6)
                win32gui.ExtTextOut(hdc, text_x, text_y, 0, None, text, None)

            win32gui.EndPaint(hwnd, paint_struct)
            return 0
        elif msg == win32con.WM_CLOSE:
            win32gui.DestroyWindow(hwnd)
            return 0
        else:
            return win32gui.DefWindowProc(hwnd, msg, wParam, lParam)

    wc = win32gui.WNDCLASS()
    wc.lpfnWndProc = wnd_proc
    wc.lpszClassName = 'MyCsWindowClass'
    
    try:
        win32gui.RegisterClass(wc)
    except win32gui.error as e:
        if e.winerror == 1410:  # Class already exists
            win32gui.UnregisterClass(wc.lpszClassName, None)
            win32gui.RegisterClass(wc)
        else:
            raise

    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

    style = win32con.WS_POPUP | win32con.WS_VISIBLE
    ex_style = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOOLWINDOW
    overlay_hwnd = win32gui.CreateWindowEx(
        ex_style,
        'MyCsWindowClass', '', style,
        0, 0, screen_width, screen_height, 0, 0, 0, None)
    
    # Make the window fully transparent
    win32gui.SetLayeredWindowAttributes(overlay_hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
    win32gui.SetWindowPos(overlay_hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    return

def update_overlay_globally(arrows):
    global current_arrows, overlay_hwnd

    if overlay_hwnd is None:
        create_overlay_window_globally()

    current_arrows = arrows
    # Invalidate the window to trigger a repaint
    win32gui.InvalidateRect(overlay_hwnd, None, True)
    win32gui.UpdateWindow(overlay_hwnd)

def make_window_invisible(hwnd):
    current_style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    new_style = current_style & ~win32con.WS_VISIBLE
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, new_style)
    win32gui.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                          win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)


def make_window_visible(hwnd):
    current_style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    new_style = current_style | win32con.WS_VISIBLE
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, new_style)
    win32gui.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                          win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)

def show_overlay(duration):
    global overlay_hwnd

    if overlay_hwnd is None:
        create_overlay_window_globally()

    make_window_visible(overlay_hwnd)

    start_time = time.time()
    while time.time() - start_time < duration:
        win32gui.PumpWaitingMessages()

    make_window_invisible(overlay_hwnd)

def is_valid_fen(fen):       
    board = chess.Board(fen)
    kingsOTB = board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None
    if not kingsOTB:
        print("is_valid_fen(): No kings")
        return False
    if board.is_game_over():
        print("is_valid_fen(): Game over")
        return False
    
    if not board.is_valid():
        print("is_valid_fen(): Board invalid")
        return False
    
    return True
  
def analyze_fen(fen):
    board = chess.Board(fen)
    global engine_analysis_time
    global engine_multi_pv
    
    moves = []
    analysis = engine.analyse(board, chess.engine.Limit(time=engine_analysis_time), multipv=engine_multi_pv) #game="new", 
    
    def get_score(variation):
        score = variation["score"].relative
        if score.is_mate():
            return float('inf') if score.mate() > 0 else float('-inf')
        return score.score()
    
    best_score = get_score(max(analysis, key=lambda x: get_score(x)))
    
    for variation in analysis:
        score = get_score(variation)
        move = variation["pv"][0]
        if (score == float('inf') and best_score == float('inf')) or (score == float('-inf') and best_score == float('-inf')):
            centipawn_loss = 0
        else:
            centipawn_loss = best_score - score
        
        moves.append({"score": score / 100.0, "centipawnloss": centipawn_loss, "squares": str(move)})
    
    return moves

def AnalyzeAndDrawMoves(fen, invertedFen, x_abs, y_abs, square_size):
    global previousFEN
    global previousMoves
    global CurrentIterationUserColor
    global UserColorChanged
    
    global draw_moves_PARAM_moves
    global draw_moves_PARAM_x_abs
    global draw_moves_PARAM_y_abs
    global draw_moves_PARAM_square_size
    
    global draw_moves_thread

    draw_moves_PARAM_square_size = square_size
    draw_moves_PARAM_x_abs = x_abs
    draw_moves_PARAM_y_abs = y_abs
    #print(f'Square Size={draw_moves_PARAM_square_size}')
    
    if fen == previousFEN and not UserColorChanged:
        draw_moves_reuse_hwnd()
        return
    #print("New FEN!")
    moves = []
    fenMoves = None
    invertedFenMoves = None
    
    if(CurrentIterationUserColor == chess.WHITE):
        fenMoves = analyze_fen(fen)
        if fenMoves is not None:
            moves += fenMoves
    else:
        invertedFenMoves = analyze_fen(invertedFen)
        if invertedFenMoves is not None:
            moves += invertedFenMoves

    previousFEN = fen
    previousMoves = moves
    UserColorChanged = False
    draw_moves_PARAM_moves = moves
    draw_moves_reuse_hwnd()
    return

def invert_active_color(fen):
    fen_parts = fen.split(' ')
    
    if len(fen_parts) != 6:
        raise ValueError("invert_active_color(): Invalid FEN string")
    
    active_color = fen_parts[1]
    fen_parts[1] = 'b' if active_color == 'w' else 'w'
    
    inverted_fen = ' '.join(fen_parts)
    return inverted_fen
    
def add_castling_rights(fen, K, Q, k, q):
    if not K and not Q and not k and not q:
        return fen
    parts = fen.split(' ')
    
    if len(parts) < 4:
        raise ValueError("add_castling_rights(): Invalid FEN format")
    
    castling_rights = ""
    
    if K:
        castling_rights += "K"
    
    if Q:
        castling_rights += "Q"
        
    if k:
        castling_rights += "k"
    
    if q:
        castling_rights += "q"
    
    parts[2] = castling_rights
    new_fen = ' '.join(parts)
    return new_fen

def check_white_castling_rights(board):
    K = True
    Q = True
    kingSquare = board.piece_at(chess.E1)
    if not kingSquare or not kingSquare.piece_type == chess.KING:
        K = False
        Q = False
        return K, Q
    
    kRookSquare = board.piece_at(chess.H1)
    if not kRookSquare or not kRookSquare.piece_type == chess.ROOK:
        K = False
    
    qRookSquare = board.piece_at(chess.A1)
    if not qRookSquare or not qRookSquare.piece_type == chess.ROOK:
        Q = False
        
    return K, Q

def check_black_castling_rights(board):
    k = True
    q = True
    kingSquare = board.piece_at(chess.E8)
    if not kingSquare or not kingSquare.piece_type == chess.KING:
        k = False
        q = False
        return k, q
    
    kRookSquare = board.piece_at(chess.H8)
    if not kRookSquare or not kRookSquare.piece_type == chess.ROOK:
        k = False
    
    qRookSquare = board.piece_at(chess.A8)
    if not qRookSquare or not qRookSquare.piece_type == chess.ROOK:
        q = False
        
    return k, q

def check_castling_rights(fen, K, Q, k, q):
    board = chess.Board(fen)
    K, Q = check_white_castling_rights(board)
    k, q = check_black_castling_rights(board)
    return K, Q, k, q

# jit compilation does not provide any benefit for functions like this one
def is_starting_position(fen):
    if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1":
        return True
    return False

def test_check_black_castling_rights():
    fen = "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w - - 0 1"
    board = chess.Board(fen)
    k, q = check_black_castling_rights(board)
    print(f"k:{k}, q:{q}")
    
def test_check_white_castling_rights():
    fen = "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w - - 0 1"
    board = chess.Board(fen)
    k, q = check_white_castling_rights(board)
    print(f"k:{k}, q:{q}")

def test_add_castling_rights():
    startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    print(add_castling_rights(startingFen, True, True, True, True))
    print(add_castling_rights(startingFen, False, False, False, False))
    
def measure_execution_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return result, execution_time

def test_captureExecTime():
    execution_times = []
    while True:
        result, execution_time = measure_execution_time(capture_chessboard)
        execution_times.append(execution_time)
        #print("Execution time[ms]:", execution_time)
        
        if len(execution_times) % 50 == 0:
            avg_time = statistics.mean(execution_times)
            std_dev = statistics.stdev(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            print(f"After {len(execution_times)} measurements:")
            print(f"Average execution time[ms]: {avg_time:.2f}")
            print(f"Standard deviation[ms]: {std_dev:.2f}")
            print(f"Minimum execution time[ms]: {min_time:.2f}")
            print(f"Maximum execution time[ms]: {max_time:.2f}")

def test_RecognizePiecesExecTime():
    execution_times = []
    chessboard, x_abs, y_abs, w, h = capture_chessboard()

    while True:
        square_size = w // 8
        result, execution_time = measure_execution_time(recognize_pieces, chessboard, square_size)
        
        execution_times.append(execution_time)
        
        if len(execution_times) % 1000 == 0:
            avg_time = statistics.mean(execution_times)
            std_dev = statistics.stdev(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            print(f"After {len(execution_times)} measurements:")
            print(f"Average execution time[ms]: {avg_time:.2f}")
            print(f"Standard deviation[ms]: {std_dev:.2f}")
            print(f"Minimum [ms]: {min_time:.2f}")
            print(f"Maximum [ms]: {max_time:.2f}")
            print(f"Recent [ms]: {execution_time:.2f}")

CurrentIterationUserColor = None
UserColorChanged = False

def decode_settings(settings_string):
    leading_char = settings_string[0]
    num_leading_chars = 0

    for char in settings_string:
        if char == leading_char:
            num_leading_chars += 1
        else:
            break
    
    print("Settings Pre Processing")
    print(settings_string)
    print("num_leading_chars:", num_leading_chars)
    settings_string = settings_string[num_leading_chars:]
    num_trailing_chars = (num_leading_chars // 2) - 2
    print("num_trailing_chars:", num_trailing_chars)
    if(num_trailing_chars != 0):
        settings_string = settings_string[:-num_trailing_chars]
    print("Settings Post Processing")
    print(settings_string)
    separator_dict = {}

    def update_separator_dict(separator):
        for char in separator:
            if char in separator_dict:
                separator_dict[char] += 1
            else:
                separator_dict[char] = 1

    float_pattern = re.compile(r'^\d+\.\d+')
    int_pattern = re.compile(r'^\d+')

    result = []

    while settings_string:
        float_match = float_pattern.match(settings_string)
        if float_match:
            value = float(float_match.group())
            result.append(value)
            print(value)
            settings_string = settings_string[float_match.end():]
            continue

        int_match = int_pattern.match(settings_string)
        if int_match:
            value = int(int_match.group())
            result.append(value)
            print(value)
            settings_string = settings_string[int_match.end():]
            continue

        # Extract separator until next number
        sep_end = 0
        while sep_end < len(settings_string) and not settings_string[sep_end].isdigit() and settings_string[sep_end] != '.':
            sep_end += 1

        separator = settings_string[:sep_end]
        update_separator_dict(separator)
        settings_string = settings_string[sep_end:]

    print(result)
    # settings order:
    # crop% top, bottom, left, _GAME_MODE_, right,
    # numberOfVariations, AnalysisDisplayDuration [ms], AnalysisTimePerVariation [ms]
    global crop_left, crop_right, crop_top, crop_bottom
    crop_left = float(result[2]) / 100.0
    crop_right = float(result[4]) / 100.0
    crop_top = float(result[0]) / 100.0
    crop_bottom = float(result[1]) / 100.0
    
    global engine_analysis_time
    global engine_multi_pv
    global engine_analysis_time_per_variation_seconds
    engine_analysis_time_per_variation_seconds = float(result[7]) / 1000.0
    engine_multi_pv = result[5]
    engine_analysis_time = round(engine_analysis_time_per_variation_seconds * engine_multi_pv, 3)
    global draw_duration
    draw_duration = float(result[6]) / 1000.0
    return result

testargs = "NNNNNNNNNNNNNNNNNN13|^|^7;;11.5-;-;-;2#?#?#?20;;;4#?#?120;;2.9NNNNNNN"

def main():
    global CurrentIterationUserColor
    args = None
    if len(sys.argv) < 2:
        print("No args")
        args = testargs
    else:
        args = sys.argv[1]

    decode_settings(args)
    set_screen_size()

    print(f'Engine analysis duration: {engine_analysis_time}')
    print(f'Draw duration: {draw_duration}')
    configureEngine()
    wait_for_picking_color()

    canCastle_K = False
    canCastle_Q = False
    canCastle_k = False
    canCastle_q = False
    
    recheck_castling_rights = False
    thread = threading.Thread(target=recheck_user_color__thread)
    thread.start()
    
    dbg_previousLoopTime = time.time()
    while True:
        try:
            chessboard, x_abs, y_abs, w, h = capture_chessboard()
            square_size = w // 8
            if square_size < 40 or square_size > 105:
                time.sleep(1)
                continue

            #print("Square Size: ", square_size)
            piece_positions = recognize_pieces(chessboard, square_size)
            #print(piece_positions)
            CurrentIterationUserColor = UserColor
            fen = generate_fen(piece_positions, CurrentIterationUserColor)
            invertedFen = invert_active_color(fen)

            if not is_valid_fen(fen if CurrentIterationUserColor == chess.WHITE else invertedFen):
                time.sleep(0.25)
                continue
                
            if(is_starting_position(fen)):
                fen = add_castling_rights(fen, True, True, True, True)
                invertedFen = add_castling_rights(invertedFen, True, True, True, True)
                recheck_castling_rights = True
            elif recheck_castling_rights:
                canCastle_K, canCastle_Q, canCastle_k, canCastle_q = check_castling_rights(fen, canCastle_K, canCastle_Q, canCastle_k, canCastle_q)
                if not canCastle_K and not canCastle_Q and not canCastle_k and not canCastle_q:
                    recheck_castling_rights = False
                else:
                    fen = add_castling_rights(fen, canCastle_K, canCastle_Q, canCastle_k, canCastle_q)
                    invertedFen = add_castling_rights(invertedFen, canCastle_K, canCastle_Q, canCastle_k, canCastle_q)
            
            AnalyzeAndDrawMoves(fen, invertedFen, x_abs, y_abs, square_size)
            loopDuration = (time.time() - dbg_previousLoopTime) * 1000
            print(f"Execution time for main(): {loopDuration:.2f} ms")
            dbg_previousLoopTime = time.time()
        except Exception as e:
            print("An error occurred:", str(e))
            #traceback.print_exc()  #Prints the full stack trace
            time.sleep(1)

if __name__ == "__main__":
    main()