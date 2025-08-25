#!/usr/bin/env python3
"""
Simple color-coded ASCII utilities.
"""
import sys, time, random

def c(code): return f"\033[{code}m"
RESET = c("0")
BOLD  = c("1")

# Colors
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
