import tkinter as tk
from tkinter import messagebox
import math
import copy

# ==============================================================================
# SECTION 0: CONFIGURATION (CONSTANTS)
# These variables do not change during the game.
# ==============================================================================

# The board size (Integers)
ROWS = 6
COLS = 7

# Codes for the pieces on the board (Integers)
# We use numbers because they are easy to store in a list.
EMPTY_SLOT = 0
PLAYER_HUMAN_ID = 1  # Red piece
PLAYER_AI_ID = 2     # Yellow piece

# AI Configuration
# 7 is deep enough to be smart, but fast thanks to optimization.
AI_MAX_SEARCH_DEPTH = 7  

# Scoring values (Integers)
SCORE_WIN = 1000000 
SCORE_DRAW = 0


# ==============================================================================
# SECTION 1: THE BOARD LOGIC (GAME RULES)
# This class handles the data and rules of Connect 4.
# ==============================================================================

class ConnectFourBoard:
    def __init__(self):
        # Create the board as a "List of Lists" (2D Array).
        # It represents a grid of 6 rows and 7 columns, filled with 0s initially.
        self.board = []
        for r in range(ROWS):
            # Create a row of 7 zeros
            new_row = [EMPTY_SLOT] * COLS
            self.board.append(new_row)

    def get_legal_actions(self):
        """
        Returns a list of column numbers where a player is allowed to drop a piece.
        A column is legal if the top cell (row 0) is empty.
        """
        valid_columns = []
        
        # Loop through every column index from 0 to 6
        for col_index in range(COLS):
            # Check if the top spot in this column is empty
            if self.board[0][col_index] == EMPTY_SLOT:
                valid_columns.append(col_index)
                
        return valid_columns

    def make_move(self, column_index, player_id):
        """
         drops a piece into a column. It falls to the lowest empty spot.
         Returns the (row, col) where it landed.
        """
        # First, check if the move is allowed
        allowed_moves = self.get_legal_actions()
        
        if column_index not in allowed_moves:
            return None # Move is not allowed

        # Start looking from the bottom row (5) up to the top row (0)
        # range(start, stop, step) -> range(5, -1, -1) counts 5, 4, 3, 2, 1, 0
        for row_index in range(ROWS - 1, -1, -1):
            
            # If we find an empty spot
            if self.board[row_index][column_index] == EMPTY_SLOT:
                # Place the player's piece there
                self.board[row_index][column_index] = player_id
                
                # Return the exact location where the piece landed
                return (row_index, column_index)
        
        return None # Should not happen if we checked legal actions first

    @staticmethod
    def check_win(board_data, last_row, last_col, player_id):
        """
        Checks if 'player_id' has 4 in a row starting from the last piece placed.
        """
        if last_row is None or last_col is None:
            return False

        # These are the 4 directions we need to check:
        # (row_change, col_change)
        directions_to_check = [
            (1, 0),   # Vertical (Down)
            (0, 1),   # Horizontal (Right)
            (1, 1),   # Diagonal (Down-Right)
            (1, -1)   # Diagonal (Down-Left)
        ]

        for direction in directions_to_check:
            # Unpack the direction tuple into two variables
            row_step = direction[0]
            col_step = direction[1]
            
            count_consecutive_pieces = 1  # Start with 1 (the piece we just placed)

            # 1. Check in the Positive Direction
            for i in range(1, 4): # Check next 3 spots: 1, 2, 3
                r = last_row + (row_step * i)
                c = last_col + (col_step * i)

                # Check if this new coordinate is essentially "on the board"
                if (0 <= r < ROWS) and (0 <= c < COLS):
                    # Check if the piece matches the player
                    if board_data[r][c] == player_id:
                        count_consecutive_pieces = count_consecutive_pieces + 1
                    else:
                        break # Stop counting if we hit a different piece/empty
                else:
                    break # Stop if we go off the board boundaries

            # 2. Check in the Negative (Opposite) Direction
            for i in range(1, 4): 
                r = last_row - (row_step * i)
                c = last_col - (col_step * i)

                if (0 <= r < ROWS) and (0 <= c < COLS):
                    if board_data[r][c] == player_id:
                        count_consecutive_pieces = count_consecutive_pieces + 1
                    else:
                        break
                else:
                    break
            
            # If we found 4 or more connected pieces in this direction, it's a win
            if count_consecutive_pieces >= 4:
                return True

        return False

    @staticmethod
    def get_legal_actions_for_copy(board_data):
        """A helper function identical to get_legal_actions but for a copied board."""
        valid_columns = []
        for col_index in range(COLS):
            if board_data[0][col_index] == EMPTY_SLOT:
                valid_columns.append(col_index)
        return valid_columns


# ==============================================================================
# SECTION 2: THE AI BRAIN (OPTIMIZED MINIMAX)
# This uses Alpha-Beta Pruning to be fast.
# ==============================================================================

class MinimaxAI:
    def __init__(self):
        self.max_search_depth = AI_MAX_SEARCH_DEPTH
        self.current_depth_tracker = 0 

    def get_ai_move(self, current_board_state):
        """
        Calculates the best column for the AI to drop a piece.
        """
        # Start with the worst possible score for the AI
        best_score_found = -math.inf
        best_column_choice = None
        
        # Initialize Alpha and Beta for pruning
        # Alpha: The best score the Maximizer (AI) can guarantee.
        # Beta: The best score the Minimizer (Human) can guarantee.
        alpha = -math.inf
        beta = math.inf
        
        # Create a copy of the board so we don't mess up the real game while thinking
        board_copy = copy.deepcopy(current_board_state)
        
        # Get all possible moves
        possible_moves = ConnectFourBoard.get_legal_actions_for_copy(board_copy)
        
        for col in possible_moves:
            # 1. Simulate making this move
            r, c = self._simulate_move(board_copy, col, PLAYER_AI_ID)
            
            # 2. Use recursion to see how good this move is.
            # We pass 'False' because after the AI moves, it is the Human's turn (Minimizer).
            score = self._run_minimax(
                board=board_copy, 
                depth=0, 
                is_maximizing_player=False, 
                last_pos=(r, c),
                alpha=alpha,
                beta=beta
            )
            
            # 3. Undo the move (backtrack) so we can try the next one
            self._undo_move(board_copy, r, c)
            
            # 4. If this move is better than what we found before, keep it
            if score > best_score_found:
                best_score_found = score
                best_column_choice = col
            
            # Update Alpha (Optimization)
            if best_score_found > alpha:
                alpha = best_score_found
        
        return best_column_choice

    # --- Helper Functions ---

    def _simulate_move(self, board_data, col, player_id):
        """Temporarily places a piece on a board copy."""
        for r in range(ROWS - 1, -1, -1):
            if board_data[r][col] == EMPTY_SLOT:
                board_data[r][col] = player_id
                return (r, col)
        return (None, None)

    def _undo_move(self, board_data, row, col):
        """Removes the piece to reset the board copy."""
        if row is not None:
            board_data[row][col] = EMPTY_SLOT

    def _calculate_score(self, board_data, last_pos, player_id):
        """
        The Heuristic Function.
        Decides how good a board state is.
        """
        # Access the current depth
        depth = self.current_depth_tracker

        # 1. Check if someone won
        did_win = ConnectFourBoard.check_win(board_data, last_pos[0], last_pos[1], player_id)
        
        if did_win:
            if player_id == PLAYER_AI_ID:
                # AI Won! Return a huge positive number.
                # We add 'MAX_DEPTH - depth' to prefer winning faster (shallower depth).
                return SCORE_WIN + (self.max_search_depth - depth)
            else:
                # Human Won! Return a huge negative number.
                # We subtract 'MAX_DEPTH - depth' to prefer losing slower (deeper depth).
                return -SCORE_WIN - (self.max_search_depth - depth)
        
        # 2. If no one won, calculate a positional score
        score = 0
        center_column_index = COLS // 2 # The middle column is index 3
        
        # Controlling the center is good strategy in Connect 4
        for r in range(ROWS):
            piece_at_center = board_data[r][center_column_index]
            
            if piece_at_center == PLAYER_AI_ID:
                score = score + 3  # Points for AI piece in center
            elif piece_at_center == PLAYER_HUMAN_ID:
                score = score - 3  # Negative points for Human piece in center
                
        return score

    # --- The Main Recursive Function ---

    def _run_minimax(self, board, depth, is_maximizing_player, last_pos, alpha, beta):
        """
        The recursive algorithm.
        is_maximizing_player = True means it is the AI's turn.
        is_maximizing_player = False means it is the Human's turn.
        """
        self.current_depth_tracker = depth 

        # Figure out who made the LAST move that got us here
        if is_maximizing_player == True:
            # If it is currently MAX's turn, then MIN (Human) just moved.
            player_who_just_moved = PLAYER_HUMAN_ID
        else:
            # If it is currently MIN's turn, then MAX (AI) just moved.
            player_who_just_moved = PLAYER_AI_ID
        
        # --- STOPPING CONDITIONS (BASE CASES) ---
        
        # 1. Check if the game is over (Win/Loss)
        is_game_over = ConnectFourBoard.check_win(board, last_pos[0], last_pos[1], player_who_just_moved)
        if is_game_over:
            return self._calculate_score(board, last_pos, player_who_just_moved)

        # 2. Check if we reached the thinking limit (Depth)
        if depth == self.max_search_depth:
            return self._calculate_score(board, last_pos, player_who_just_moved)

        # 3. Check for Draw (Board full)
        valid_moves = ConnectFourBoard.get_legal_actions_for_copy(board)
        if len(valid_moves) == 0:
            return SCORE_DRAW
        
        # --- RECURSION STEPS ---
        
        if is_maximizing_player: 
            # AI's Turn: Wants to MAXIMIZE the score
            max_value = -math.inf
            
            for col in valid_moves:
                r, c = self._simulate_move(board, col, PLAYER_AI_ID)
                
                # Recursion: Call self, but increase depth and switch turn to False (Human)
                current_value = self._run_minimax(board, depth + 1, False, (r, c), alpha, beta)
                
                # Keep the highest score found
                if current_value > max_value:
                    max_value = current_value
                
                self._undo_move(board, r, c)
                
                # *** ALPHA-BETA PRUNING ***
                # Update Alpha (Best MAX path found so far)
                if max_value > alpha:
                    alpha = max_value
                
                # If Alpha is better than Beta, the Minimizer will never let us get here.
                # So we stop looking (Prune).
                if alpha >= beta:
                    break 
            
            return max_value
            
        else: 
            # Human's Turn: Wants to MINIMIZE the score (make it negative)
            min_value = math.inf
            
            for col in valid_moves:
                r, c = self._simulate_move(board, col, PLAYER_HUMAN_ID)
                
                # Recursion: Call self, but increase depth and switch turn to True (AI)
                current_value = self._run_minimax(board, depth + 1, True, (r, c), alpha, beta)
                
                # Keep the lowest score found
                if current_value < min_value:
                    min_value = current_value
                
                self._undo_move(board, r, c)
                
                # *** ALPHA-BETA PRUNING ***
                # Update Beta (Best MIN path found so far)
                if min_value < beta:
                    beta = min_value
                
                # If Alpha is better than Beta, the Maximizer has a better option elsewhere.
                # So we stop looking (Prune).
                if alpha >= beta:
                    break
            
            return min_value


# ==============================================================================
# SECTION 3: THE USER INTERFACE (GUI)
# This handles the window, buttons, and mouse clicks using Tkinter.
# ==============================================================================

class ConnectFourGUI:
    def __init__(self, root_window):
        self.master = root_window
        self.master.title("Connect 4 - AI Project")
        
        # Create the logic objects
        self.board_logic = ConnectFourBoard()
        self.ai_agent = MinimaxAI() 
        
        # Game State Variables
        self.game_mode = None          # Will be 'AI' or 'Human'
        self.active_player = PLAYER_HUMAN_ID
        self.is_game_running = False
        
        # Variables to remember the last move (for win checking)
        self.last_row = None
        self.last_col = None
        
        # A "Lock" variable. This prevents the user from clicking twice 
        # while the AI is thinking.
        self.is_processing_move = False 

        # Graphics Settings
        self.SQUARE_SIZE = 80
        width_px = COLS * self.SQUARE_SIZE
        height_px = ROWS * self.SQUARE_SIZE
        
        # Create the container frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(pady=10, padx=10)
        
        # Start by showing the menu
        self.show_start_menu()

    def clear_screen(self):
        """Removes all buttons and text from the screen."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_start_menu(self):
        """Shows the two buttons to pick the game mode."""
        self.clear_screen()
        
        # Title Text
        title = tk.Label(self.main_frame, text="Select Game Mode", font=('Arial', 20, 'bold'))
        title.pack(pady=30)
        
        # Button 1: Human vs AI
        btn_ai = tk.Button(self.main_frame, text="Human vs. AI", 
                           command=lambda: self.initialize_game('AI'),
                           bg='#2196F3', fg='white', font=('Arial', 14, 'bold'), 
                           width=30, height=2)
        btn_ai.pack(pady=15)
                  
        # Button 2: Human vs Human
        btn_human = tk.Button(self.main_frame, text="Human vs. Human", 
                              command=lambda: self.initialize_game('Human'),
                              bg='#FF9800', fg='white', font=('Arial', 14, 'bold'), 
                              width=30, height=2)
        btn_human.pack(pady=15)

    def show_game_board(self):
        """Sets up the visual board grid."""
        self.clear_screen()
        
        # Create a separate frame for the game elements
        self.game_frame = tk.Frame(self.main_frame)
        self.game_frame.pack()
        
        # Status Text (e.g. "Player 1 Turn")
        self.status_label = tk.Label(self.game_frame, text="", font=('Arial', 14))
        self.status_label.pack(pady=5)
        
        # The Canvas is where we draw the circles
        w = COLS * self.SQUARE_SIZE
        h = ROWS * self.SQUARE_SIZE
        self.canvas = tk.Canvas(self.game_frame, width=w, height=h, bg='blue')
        self.canvas.pack(pady=10)
        
        # Connect the mouse click event to our function
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Restart Button
        btn_restart = tk.Button(self.game_frame, text="Restart / Change Mode", 
                                command=self.reset_game, 
                                bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'))
        btn_restart.pack(pady=10)

    def initialize_game(self, selected_mode):
        """Sets up the variables to start a new game."""
        self.game_mode = selected_mode
        self.board_logic = ConnectFourBoard() # Reset board data
        self.active_player = PLAYER_HUMAN_ID
        self.is_game_running = True
        self.is_processing_move = False # Unlock the mouse
        
        self.show_game_board()
        
        if self.game_mode == 'AI':
            self.status_label.config(text="Human (Red) vs. AI (Yellow). Your Turn!")
        else:
            self.status_label.config(text="Player 1 (Red) vs. Player 2 (Yellow). P1's Turn!")

        self.redraw_graphics()

    def redraw_graphics(self):
        """Erases the canvas and redraws all circles based on board data."""
        self.canvas.delete("all")
        
        # Loop through every cell in the grid
        for r in range(ROWS):
            for c in range(COLS):
                # Calculate pixel coordinates
                x1 = c * self.SQUARE_SIZE
                y1 = r * self.SQUARE_SIZE
                x2 = x1 + self.SQUARE_SIZE
                y2 = y1 + self.SQUARE_SIZE
                
                # Draw the blue background circle (the empty hole look)
                self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, 
                                        fill="lightblue", outline="darkblue")
                
                # Check who owns this spot
                piece_id = self.board_logic.board[r][c]
                
                color_to_fill = None
                if piece_id == PLAYER_HUMAN_ID:
                    color_to_fill = "red"
                elif piece_id == PLAYER_AI_ID:
                    color_to_fill = "yellow"
                
                # If there is a piece, draw it on top
                if color_to_fill is not None:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, 
                                            fill=color_to_fill, outline="black", width=2)

    def on_canvas_click(self, event):
        """This runs when the user clicks the mouse on the board."""
        
        # --- BUG FIX: INPUT LOCK ---
        # If the game is over OR the computer is currently thinking, ignore the click.
        if self.is_game_running == False:
            return
        if self.is_processing_move == True:
            return
            
        # Lock the input so the user can't click again immediately
        self.is_processing_move = True 

        # Figure out which column was clicked
        pixel_x = event.x
        column_clicked = pixel_x // self.SQUARE_SIZE
        
        # Check if this column is valid
        legal_moves = self.board_logic.get_legal_actions()
        
        if column_clicked in legal_moves:
            # 1. Perform the Human's Move
            result = self.board_logic.make_move(column_clicked, self.active_player)
            
            # Store the row and col for checking wins
            self.last_row = result[0]
            self.last_col = result[1]
            
            self.redraw_graphics()
            
            # 2. Check if Human Won
            game_ended = self.check_game_over(self.active_player)
            if game_ended == True:
                self.is_processing_move = False # Unlock so they can click restart
                return
            
            # 3. Switch Turns
            if self.game_mode == 'Human':
                # Logic for 2-Player Mode
                if self.active_player == PLAYER_HUMAN_ID:
                    self.active_player = PLAYER_AI_ID # Player 2
                    name = "Player 2 (Yellow)"
                else:
                    self.active_player = PLAYER_HUMAN_ID # Player 1
                    name = "Player 1 (Red)"
                
                self.status_label.config(text=f"{name}'s Turn")
                self.is_processing_move = False # Unlock immediately for next human
            
            elif self.game_mode == 'AI':
                # Logic for AI Mode
                self.active_player = PLAYER_AI_ID
                self.status_label.config(text="AI is Thinking...")
                
                # Force the window to update so the text shows up
                self.master.update() 
                
                # Run the AI logic after a tiny delay (100ms) so graphics update first
                self.master.after(100, self.run_ai_turn) 
        else:
            # If they clicked a full column, just unlock and let them try again
            self.is_processing_move = False 


    def check_game_over(self, player_id):
        """Checks for Win or Draw. Returns True if game ended."""
        # Check Win
        has_won = ConnectFourBoard.check_win(self.board_logic.board, self.last_row, self.last_col, player_id)
        
        if has_won:
            winner_text = ""
            if self.game_mode == 'AI':
                if player_id == PLAYER_AI_ID:
                    winner_text = "AI Player (Yellow) Wins!"
                else:
                    winner_text = "Human Player (Red) Wins!"
            else:
                if player_id == PLAYER_HUMAN_ID:
                    winner_text = "Player 1 (Red) Wins!"
                else:
                    winner_text = "Player 2 (Yellow) Wins!"
            
            self.trigger_game_over(winner_text)
            return True

        # Check Draw (No legal moves left)
        available = self.board_logic.get_legal_actions()
        if len(available) == 0:
            self.trigger_game_over("It's a Draw!")
            return True
            
        return False

    def run_ai_turn(self):
        """This function runs the AI logic."""
        # Double check lock
        if self.is_game_running == False:
            self.is_processing_move = False
            return

        # 1. Ask Minimax for the best column
        # We send a copy of the board data
        board_data_copy = [row[:] for row in self.board_logic.board] 
        best_col = self.ai_agent.get_ai_move(board_data_copy)
        
        # 2. Make the AI move
        result = self.board_logic.make_move(best_col, PLAYER_AI_ID)
        self.last_row = result[0]
        self.last_col = result[1]
        
        self.redraw_graphics()
        
        # 3. Check if AI won
        game_ended = self.check_game_over(PLAYER_AI_ID)
        if game_ended:
            self.is_processing_move = False
            return
            
        # 4. Switch back to Human
        self.active_player = PLAYER_HUMAN_ID
        self.status_label.config(text="Your Turn (Red)")
        
        # CRITICAL: Unlock the input now that AI is done
        self.is_processing_move = False 
        
    def trigger_game_over(self, message_text):
        """Freezes the game and shows a popup."""
        self.is_game_running = False
        self.status_label.config(text=message_text, fg='red', font=('Arial', 16, 'bold'))
        messagebox.showinfo("Game Over", message_text)

    def reset_game(self):
        """Goes back to the start menu."""
        self.is_game_running = False
        self.game_mode = None
        self.is_processing_move = False
        self.show_start_menu()


# ==============================================================================
# MAIN EXECUTION
# This block runs when you double-click the python file.
# ==============================================================================
if __name__ == '__main__':
    # Create the main window
    main_window = tk.Tk()
    
    # Start the game logic
    game_app = ConnectFourGUI(main_window)
    
    # Keep the window open
    main_window.mainloop()