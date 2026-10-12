import os
import pyautogui
import time
import random
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets, LogLevels
from brainflow.data_filter import DataFilter
from playsound import playsound


BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.ip_port = 6789
params.ip_address = "192.168.4.1"
board = BoardShim(BoardIds.CYTON_DAISY_WIFI_BOARD, params)

board.prepare_session()

# Check sample rate, change to 250 Hz
board.config_board("~~")
board.config_board("~6")

# Check board mode, change to Marker mode
board.config_board("//")
board.config_board("/4")


for i in range(1,9):
    board.config_board("x" + str(i) + "000000X")

daisy_channels = "QWERTYUI"

for i in range(0,8):
    board.config_board("x" + daisy_channels[i] + "000000X")

num_tests = 88; # 2 blocks for 44 phonemes
wait_slide = '47' # slide number of wait slide
begin_test_slide = '2' # slide number of begin test slide


# Corresponds to slides 3-4647

slides =[
    ['i_colon', 2],
    ['I', 2],
    ['hsu', 2],
    ['bar_u', 2],
    ['e', 2],
    ['rot_e', 2],
    ['uhr', 2],
    ['oa', 2],
    ['ae', 2],
    ['arch', 2],
    ['ar', 2],
    ['rot_c', 2],
    ['ie', 2],
    ['ei', 2],
    ['omega_e', 2],
    ['ci', 2],
    ['e_omega', 2],
    ['ee', 2],
    ['ai', 2],
    ['a_omega', 2],
    ['p', 2],
    ['f', 2],
    ['t', 2],
    ['theta', 2],
    ['t_integral', 2],
    ['s', 2],
    ['integral', 2],
    ['k', 2],
    ['b', 2],
    ['v', 2],
    ['d', 2],
    ['eth', 2],
    ['d3', 2],
    ['z', 2],
    ['3', 2],
    ['g', 2],
    ['h', 2],
    ['m', 2],
    ['n', 2],
    ['nn', 2],
    ['r', 2],
    ['l', 2],
    ['w', 2],
    ['j', 2],
    ]

# Tests if each block was repeated 2 times for 10 total tests per phoneme
test_slides =[
    ['i_colon', 0],
    ['I', 0],
    ['hsu', 0],
    ['bar_u', 0],
    ['e', 0],
    ['rot_e', 0],
    ['uhr', 0],
    ['oa', 0],
    ['ae', 0],
    ['arch', 0],
    ['ar', 0],
    ['rot_c', 0],
    ['ie', 0],
    ['ei', 0],
    ['omega_e', 0],
    ['ci', 0],
    ['e_omega', 0],
    ['ee', 0],
    ['ai', 0],
    ['a_omega', 0],
    ['p', 0],
    ['f', 0],
    ['t', 0],
    ['theta', 0],
    ['t_integral', 0],
    ['s', 0],
    ['integral', 0],
    ['k', 0],
    ['b', 0],
    ['v', 0],
    ['d', 0],
    ['eth', 0],
    ['d3', 0],
    ['z', 0],
    ['3', 0],
    ['g', 0],
    ['h', 0],
    ['m', 0],
    ['n', 0],
    ['nn', 0],
    ['r', 0],
    ['l', 0],
    ['w', 0],
    ['j', 0],
    ]

# Opens the powerpoint
# Cole - File Path
#fn = r"C:\Users\dchel\source\repos\StimPres\StimPres\stimulus.pptx"
# Jason - File Path
#fn = "Users/jason/Documents/Github/StimPres/stimulus.pptx"
# Sam - File Path
#fn = "C:\GitHub\GhostTalker\StimPres\stimulus.pptx"


absolute_path = os.path.dirname(__file__)
relative_path = "stimulus.pptx"
full_path = os.path.join(absolute_path, relative_path)

#Mac OS Start
#os.start("name" + full_path) # Potential Correct MAC Start

# PC OS Start
os.startfile(full_path)

time.sleep(2)

# Sets the powerpoint to fullscreen
pyautogui.hotkey(full_path,'f5')

#time.sleep(5)
# TODO Implement the Rules slide
time.sleep(2)
# Move to the Rules Slide
pyautogui.press('enter')
# Sleep to allow reading of the rules
time.sleep(7)

# one_block presents visual and auditory stimulus, then conducts five EEG tests separated into 2 second intervals
# slides_place is the index in the slides of the phoneme being tested
def one_block(slides_place):
    # phoneme slides with audio begin at slide 3
    current_audio_slide = slides_place + 3
    str_current_audio_slide = str(current_audio_slide)

    #phoneme slides without audio begin at slide 48
    current_slide = slides_place + 48
    str_current_slide = str(current_slide)

    #audio file path
    audio_path = 'slide' + str(current_audio_slide) + '.mp3'
    full_audio_path = os.path.join(absolute_path, audio_path)

    # keypress for the slide number, if slide number has two digits press the digit in the ones place
    pyautogui.press(str_current_audio_slide[0])
    if(current_audio_slide >= 10):
        pyautogui.press(str_current_audio_slide[1])
    pyautogui.press('enter') 

    playsound(full_audio_path)
    
    time.sleep(2)
    
    board.start_stream()
    # repeat test procedure 5 times
    # Test Procedure Change from 6 to 31
    for i in range(1,6):

        # test_slides incremented by 1
        test_slides[slides_place][1] += 1

        #load wait slide
        pyautogui.press(wait_slide[0])
        pyautogui.press(wait_slide[1])
        pyautogui.press('enter')

        # 1 second buffer
        time.sleep(1)

        # load phoneme slide
        pyautogui.press(str_current_slide[0])
        if(current_slide >= 10):
            pyautogui.press(str_current_slide[1])
        pyautogui.press('enter')
        board.insert_marker(i)
        
        time.sleep(2)
        board.insert_marker(i)

        # 2 second buffer
        #time.sleep(2)               
    #load wait slide
    pyautogui.press(wait_slide[0])
    pyautogui.press(wait_slide[1])
    pyautogui.press('enter')

    data = board.get_board_data()
    board.stop_stream()
    
    ## IMPORTANT Change Initials for Each Test Subject
    initials = 'GT007'

    # Old Naming Convention:
    # naming_convention = initials + str(test_slides[slides_place][1]) + '_' + slides[slides_place][0]
    # Background Naming Convention
    # naming_convention = initials + '_' + str(slides_place) +'_BKGRD'
    # Current Naming Convention - NOTE DLR's 8th test currently
    naming_convention = initials + '_' + str(slides_place) +'_3'

    # Print File
    DataFilter.write_file(data, naming_convention + '.txt', 'w')


    # 2 second buffer
    time.sleep(2)

    return

# generates random phoneme and tests it if it has not reached the trial limit (10 trials for each phoneme)
while num_tests > 0:

    # pseudo-randomly generated number in range for all phonemes
    slides_place = random.randint(0, 43)

    # test the phoneme if slides counter has not reached zero
    if(slides[slides_place][1] > 0):
        slides[slides_place][1] -= 2
        num_tests -= 2
        one_block(slides_place)

board.release_session()
    
#prints number of blocks conducted for every phoneme
for i in range(0,43):

    print(test_slides[i][1])

pyautogui.hotkey('esc')