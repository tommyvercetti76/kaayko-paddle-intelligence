#!/usr/bin/env python3
"""
🎨 ANSI Color and Formatting Utilities
Beautiful terminal colors for the KAAYKO collection system
"""
import sys, time, random

def c(code): return f"\033[{code}m"
RESET = c("0")
BOLD  = c("1")

# Legacy colors (keeping for compatibility)
RED    = c("91")
GREEN  = c("92")
YELLOW = c("93")
BLUE   = c("94")
MAGENTA= c("95")
CYAN   = c("96")
GRAY   = c("90")
WHITE  = c("97")

def ok(msg):     print(f"{GREEN}✔{RESET} {msg}")
def warn(msg):   print(f"{YELLOW}⚠{RESET} {msg}")
def err(msg):    print(f"{RED}✖{RESET} {msg}")
def info(msg):   print(f"{CYAN}➤{RESET} {msg}")
def step(msg):   print(f"{MAGENTA}▸{RESET} {msg}")
def title(msg):  print(f"{BOLD}{WHITE}{msg}{RESET}")
def sect(msg):   print(f"\n{BOLD}{BLUE}=== {msg} ==={RESET}")

class C:
    """ANSI color codes for beautiful terminal output."""
    
    # Colors
    RED = '\033[91m'
    GRN = '\033[92m'      # Green
    YEL = '\033[93m'      # Yellow
    BLU = '\033[94m'      # Blue
    MAG = '\033[95m'      # Magenta
    CYA = '\033[96m'      # Cyan
    WHT = '\033[97m'      # White
    GRY = '\033[90m'      # Gray
    
    # Styles
    B = '\033[1m'         # Bold
    U = '\033[4m'         # Underline
    I = '\033[3m'         # Italic
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GRN = '\033[102m'
    BG_YEL = '\033[103m'
    BG_BLU = '\033[104m'
    BG_MAG = '\033[105m'
    BG_CYA = '\033[106m'
    BG_WHT = '\033[107m'
    BG_GRY = '\033[100m'
    
    # Reset
    R = '\033[0m'         # Reset all formatting

def colorize(text, color):
    """Apply color to text with automatic reset."""
    return f"{color}{text}{C.R}"

def success(text):
    """Format success message."""
    return f"{C.GRN}✅ {text}{C.R}"

def error(text):
    """Format error message."""
    return f"{C.RED}❌ {text}{C.R}"

def warning(text):
    """Format warning message."""
    return f"{C.YEL}⚠️  {text}{C.R}"

def info_msg(text):
    """Format info message."""
    return f"{C.CYA}ℹ️  {text}{C.R}"

def progress_bar(current, total, width=50):
    """Create a colored progress bar."""
    if total == 0:
        return f"{C.GRY}{'─' * width}{C.R} 0%"
    
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    
    # Color based on progress
    if percent < 0.3:
        color = C.RED
    elif percent < 0.7:
        color = C.YEL
    else:
        color = C.GRN
    
    return f"{color}{bar}{C.R} {percent*100:.1f}%"

def header(text, char='═', width=80):
    """Create a beautiful header."""
    return f"{C.B}{C.CYA}{char * width}{C.R}\n{C.B}{text.center(width)}{C.R}\n{C.B}{C.CYA}{char * width}{C.R}"

def box(text, padding=2):
    """Put text in a box."""
    lines = text.split('\n')
    max_width = max(len(line) for line in lines) + padding * 2
    
    result = f"{C.CYA}┌{'─' * max_width}┐{C.R}\n"
    for line in lines:
        padded_line = f"{' ' * padding}{line}{' ' * (max_width - len(line) - padding)}"
        result += f"{C.CYA}│{C.R}{padded_line}{C.CYA}│{C.R}\n"
    result += f"{C.CYA}└{'─' * max_width}┘{C.R}"
    
    return result
