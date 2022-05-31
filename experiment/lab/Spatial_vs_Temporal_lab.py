"""                                                        ^
                                                           ^
                                                           ^
                                                           ^
PRESS PLAY BUTTON TO RUN EXPERIMENT >   >   >   >   >   >  /
select conditions in pseudorandom order
complete 5 runs in each condition

Web vs LAB Data Collection



This is NOT for final year project students
    - contact Richard at lpxrl4@nottingham.ac.uk for help

"""


from __future__ import absolute_import, division

import copy
import datetime
import pandas as pd
import pyglet
from psychopy import locale_setup, monitors
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
from random import choice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard


# ---------- Define Functions ---------- #
def get_orientation(ori_):
    if ori_ == 'vertical':
        ori_ = 0
    elif ori_ == 'plus45':
        ori_ = 45  # *-1 as buffer flips image
    elif ori_ == 'horizontal':
        ori_ = 90
    elif ori_ == 'minus45':
        ori_ = -45  # *-1 as buffer flips image
    return ori_


class PseudorandomOrder:
    # randomly chooses item from list whilst ensuring each item shows equal n times
    def __init__(self, items, block_size=5):
        self.items = items  # e.g. [['f','j'], ['j','f']
        self.block_size = block_size  # n repeats before resetting
        self.current_order = []
        self.idx_keys = [f"{i}" for i in range(len(items))]
        self.item_counter = {}
        for i in self.idx_keys:
            self.item_counter[i] = 0  # init counter for each order

    def get_current_order(self, update=False):
        if update:
            # randomly choose idx_key as long as counter is < block_size
            idx_choice = choice([i for i in self.idx_keys if self.item_counter[i] < self.block_size])
            self.current_order = self.items[int(idx_choice)]  # set current_order by indexing into self.order_list
            self.item_counter[idx_choice] += 1  # update counter
            # if all reached block_size reset counters
            if all(self.item_counter[i] == self.block_size for i in self.idx_keys):
                for i in self.idx_keys:
                    self.item_counter[i] = 0
        return self.current_order


class LevelHandler:
    def __init__(self, std_ori, ori_offset, max_count=10):
        self.std_ori = std_ori
        self.ori_offset = ori_offset
        self.max_count = max_count  # n_reps of levels
        self.current_count = 0
        self.n_correct = 0
        self.proportion_correct = None

    def use_level(self):
        if self.current_count < self.max_count:
            self.current_count += 1  # add to counter when trial ori is set
            return self.ori_offset
        else:
            return None  # do not return ori if max_count reached

    def calc_proportion_correct(self):
        self.proportion_correct = (1 / self.current_count) * self.n_correct
        return self.proportion_correct


class MocsHandler:
    def __init__(self, std_ori, ori_offsets, n_reps):
        self.std_ori = std_ori
        self.ori_offsets = ori_offsets  # all possible ori_offset levels
        self.n_reps = n_reps  # n_reps of each level
        self.all_levels = [LevelHandler(std_ori, i_offset, n_reps) for i_offset in ori_offsets]
        self.current_level = None
        self.continue_trials = True

    def set_current_level(self):
        possible_levels = []  # possible levels for present trial
        for i_level in self.all_levels:
            if i_level.current_count < self.n_reps:
                possible_levels.append(i_level)
        self.current_level = choice(possible_levels)
        self.check_if_end()

    def get_trial_level(self):
        self.set_current_level()  # randomise level for this trial
        self.check_if_end()
        return self.current_level.use_level()

    def update_correct(self, is_correct):
        if not any(is_correct == i for i in [0, 1]):
            raise TypeError('0 for incorrect\n1 for correct')
        else:
            self.current_level.n_correct += is_correct
        self.check_if_end()
        # end pilot_data when all levels shown 10 times

    def check_if_end(self):
        if all(i_level.current_count >= self.n_reps for i_level in self.all_levels):
            self.continue_trials = False
            return not self.continue_trials

    def get_results(self):
        results = dict(offset=[], proportion_correct=[])
        # calc proportion correct for each level
        for i_level in self.all_levels:
            results['offset'].append(str(i_level.ori_offset))
            results['proportion_correct'].append(i_level.calc_proportion_correct())
        return pd.DataFrame(results)


def check_if_fin(data_directory, p_, o_, t_, nmax):
    # check whether selected conds have been done >nmax times
    ogdir = os.getcwd()
    os.chdir(data_directory)
    counter = 0
    for file in os.listdir():
        if file.endswith('.csv'):
            if p_ == file.split('_')[4] and \
               t_ == file.split('_')[6] and \
               o_ == file.split('_')[5]:
                counter += 1
    os.chdir(ogdir)
    if counter >= nmax:
        return True
    else:
        return False


def count_runs(data_directory):
    runs = {'p': [], 'o': [], 't': []}
    og_directory = os.getcwd()
    os.chdir(data_directory)
    for file in os.listdir():
        if file.endswith('.csv'):
            if "".join(file.split('_')[0:3]) != 'SpatialvsTemporal':
                continue
            runs['p'].append(file.split('_')[4])
            runs['o'].append(file.split('_')[6])
            runs['t'].append(file.split('_')[5])
    runs = pd.DataFrame(runs)
    os.chdir(og_directory)
    runs = runs.sort_values(by=['p', 'o', 't'])
    runs.to_csv('nruns.csv')
    return runs


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
ogdir = os.getcwd()
datadir = os.path.join(ogdir, 'data')

participants = ['dm', 'test']
orientations = ['vertical', 'horizontal', 'plus45', 'minus45']
tasks = ['spatial2AFC', 'temporal2AFC']
# Store info about the pilot_data session
expName = 'Spatial_vs_Temporal_lab'  # from the Builder filename that created this script
expInfo = {'participant': participants,
           'orientation': orientations,
           'task': tasks}
dlg_open = True
while dlg_open:
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if not dlg.OK:
        core.quit()  # user pressed cancel
    else:
        expInfo['analysis'] = copy.copy(expInfo)
        dlg_open = False

# use info from gui input
tStamp = datetime.datetime.now()  # create timestamp
expInfo['date'] = f'{tStamp.day}d{tStamp.month}m{tStamp.year}y{tStamp.hour}h{tStamp.minute}m'
expInfo['expName'] = expName
expInfo['mocs'] = {'offsets': [], 'n_reps': 10}
if any(expInfo['orientation'] == i for i in ['vertical', 'horizontal']):
    expInfo['mocs']['offsets'] = [0.71, 1.43, 2.14, 2.86, 3.57, 4.29, 5.00]  # cardinal

    # PCN_21 had diff range
    if any(p == expInfo['participant'] for p in ['mgd', 'mws']):
        expInfo['mocs']['offsets'] = [0.57, 1.14, 1.71, 2.29, 2.86, 3.43, 4.00]  # cardinal
    expInfo['CvsO'] = 'cardinal'
    z=0
else:
    expInfo['mocs']['offsets'] = [3.57, 7.14, 10.71, 14.29, 17.86, 21.43, 25.00]  # oblique
    expInfo['CvsO'] = 'oblique'

if expInfo['participant'] == 'test':
    expInfo['mocs']['n_reps'] = 1
    expInfo['mocs']['offsets'] = [3, 6, 12, 24]

if expInfo['participant'] == 'rjl':
    if expInfo['CvsO'] == 'cardinal':
        expInfo['mocs']['offsets'] = [0.36, 0.71, 1.07, 1.43, 1.79, 2.14, 2.5]
    else:
        expInfo['mocs']['offsets'] = [1.79, 3.57, 5.36, 7.14, 8.93, 10.71, 12.5]

data_path = os.path.join(_thisDir, 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)
filename = os.path.join(data_path, f"{expInfo['expName']}"
                                   f"_{expInfo['participant']}"
                                   f"_{expInfo['orientation']}"
                                   f"_{expInfo['task']}"
                                   f"_{expInfo['date']}")  # unique timestamp prevents overwrite

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=True, saveWideText=False, dataFileName=filename)
endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Initilise the window
screen = pyglet.canvas.Display().get_default_screen()  # get screen dimensions
win = visual.Window(
    size=(screen.width, screen.height), fullscr=True, screen=0,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True, units='norm')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] is not None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Initialise all clock components
clockInstructions = core.Clock()
clockTarget = core.Clock()
clockTrial = core.Clock()

# Initialise all keyboard components
defaultKeyboard = keyboard.Keyboard()  # default keyboard to check for escape
keyInstructions = keyboard.Keyboard()
keyTarget = keyboard.Keyboard()
keyTrial = keyboard.Keyboard()

# Initialise all text components
if expInfo['task'] == 'spatial2AFC':
    instructions_text = f"Two gratings will appear simultaneously on the left and right of the screen." \
                        f"\n\nRespond with the F key if the target orientation appears on the left.\n\n" \
                        f"Respond with the J key if the target orientation appears on the right.\n\n" \
                        f"If you are unsure then respond with your best guess." \
                        f"\n\nMaintain your fixation on the central dot at all times!\n\n" \
                        f"Press F or J to continue."

    target_text = f"'This is your target grating..." \
                  f"\n...on which side does it appear?" \
                  f"\n\npress F if left" \
                  f"\npress J if right',"

elif expInfo['task'] == 'temporal2AFC':
    instructions_text = f"'Two gratings will be presented one after the other in the centre of your screen." \
                        f"\n\nRespond with the F key if the target orientation appears in the 1st interval." \
                        f"\n\nRespond with the J key if the target orientation appears in the 2nd interval." \
                        f"\n\nIf you are unsure then respond with your best guess." \
                        f"\n\nPress F or J to continue."

    target_text = f"This is your target grating..." \
                  f"\n...in which interval does it appear?" \
                  f"\n\npress F if first interval" \
                  f"\npress J if second interval"

textInstructions = visual.TextStim(win=win, name='textInstructions', text=instructions_text, font='Arial',
                                   units='norm', pos=(0, 0), height=0.075, wrapWidth=1.9, ori=0, color='white',
                                   colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)

textTarget = visual.TextStim(win=win, name='textTarget', text=target_text, font='Arial', units='norm',
                             pos=(0, 0.7), height=0.075, wrapWidth=1.9, ori=0, color='white',
                             colorSpace='rgb', opacity=1, languageStyle='LTR')

textBegin = visual.TextStim(win=win, name='textBegin',
                            text="Maintain fixation on the central dot."
                                 "\n\npress 'F' or 'J' to begin",
                            font='Arial', units='norm', pos=(0, -0.7), height=0.075, wrapWidth=2, ori=0,
                            color='white', colorSpace='rgb', opacity=1, languageStyle='LTR')

# Initialise stimulus properties
stim = {'ori': get_orientation(expInfo['orientation']),
        'contrast': 0.4,
        'size': {
            'deg': 8,
            'pix': None},
        'offset': 8,
        'phase': None,
        'sf': {
            'cpd': 1,
            'ppc': None,
            'cpp': None},
        'mask': 'gauss',
        'res': 512,
        'duration': 0.3,
        'tstart': [0.8, 1.7]  # each interval
        }

# Initialise stimuli
gratingTarget = visual.GratingStim(win=win, units='deg', tex='sin', mask='gauss',
                                   ori=stim['ori'], pos=(0, 0), size=stim['size']['deg'], sf=stim['sf']['cpd'],
                                   color=[1, 1, 1], colorSpace='rgb', opacity=1, blendmode='avg',
                                   texRes=512, interpolate=True, contrast=stim['contrast'])

gratingFoil = visual.GratingStim(win=win, units='deg', tex='sin', mask='gauss',
                                 ori=stim['ori'], pos=(0, 0), size=stim['size']['deg'], sf=stim['sf']['cpd'],
                                 color=[1, 1, 1], colorSpace='rgb', opacity=1, blendmode='avg',
                                 texRes=512, interpolate=True, contrast=stim['contrast'])

dotFixation = visual.Polygon(win=win, name='dotFixation', units='deg', edges=32, size=[.1, .1], ori=0, pos=(0, 0),
                             lineWidth=0, lineColor=[-1] * 3, lineColorSpace='rgb', fillColor=[-.15] * 3,
                             fillColorSpace='rgb', opacity=1, interpolate=True)

# Initilise columns for output as .pkl
summary_cols = ["expname", "participant", "task", "orientation", "ori_offset", "proportion_correct"]

# Create some handy timers
globalTimer = core.Clock()  # to track the time since pilot_data started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

# ------Prepare to start Routine "instructions"-------
continueRoutine = True
# update component parameters for each repeat
keyInstructions.keys = []
keyInstructions.rt = []
_key_instructions_allKeys = []
# keep track of which components have finished
instructionsComponents = [textInstructions, keyInstructions]
for thisComponent in instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
clockInstructions.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions"-------
while continueRoutine:
    # get current time
    t = clockInstructions.getTime()
    tThisFlip = win.getFutureFlipTime(clock=clockInstructions)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (0 = first frame)
    # update/draw components on each frame
    # *textInstructions* updates
    if textInstructions.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        textInstructions.frameNStart = frameN  # exact frame index
        textInstructions.tStart = t  # local t and not account for scr refresh
        textInstructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textInstructions, 'tStartRefresh')  # time at next scr refresh
        textInstructions.setAutoDraw(True)

    # *keyInstructions* updates
    waitOnFlip = False
    if keyInstructions.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        keyInstructions.frameNStart = frameN  # exact frame index
        keyInstructions.tStart = t  # local t and not account for scr refresh
        keyInstructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(keyInstructions, 'tStartRefresh')  # time at next scr refresh
        keyInstructions.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(keyInstructions.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(keyInstructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if keyInstructions.status == STARTED and not waitOnFlip:
        theseKeys = keyInstructions.getKeys(keyList=['y', 'n', 'left', 'right', 'space', 'f', 'j'], waitRelease=False)
        _key_instructions_allKeys.extend(theseKeys)
        if len(_key_instructions_allKeys):
            keyInstructions.keys = _key_instructions_allKeys[-1].name  # just the last key pressed
            keyInstructions.rt = _key_instructions_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions"-------
for thisComponent in instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# the Routine "instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "target"-------
continueRoutine = True
# update component parameters for each repeat
keyTarget.keys = []
keyTarget.rt = []
_key_target_allKeys = []
# keep track of which components have finished
targetComponents = [textTarget, keyTarget, gratingTarget, textBegin]
for thisComponent in targetComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
clockTarget.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "target"-------
while continueRoutine:
    # get current time
    t = clockTarget.getTime()
    tThisFlip = win.getFutureFlipTime(clock=clockTarget)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *textTarget* updates
    if textTarget.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        textTarget.frameNStart = frameN  # exact frame index
        textTarget.tStart = t  # local t and not account for scr refresh
        textTarget.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textTarget, 'tStartRefresh')  # time at next scr refresh
        textTarget.setAutoDraw(True)

    # *keyTarget* updates
    waitOnFlip = False
    if keyTarget.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        keyTarget.frameNStart = frameN  # exact frame index
        keyTarget.tStart = t  # local t and not account for scr refresh
        keyTarget.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(keyTarget, 'tStartRefresh')  # time at next scr refresh
        keyTarget.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(keyTarget.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(keyTarget.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if keyTarget.status == STARTED and not waitOnFlip:
        theseKeys = keyTarget.getKeys(keyList=['y', 'n', 'left', 'right', 'space', 'f', 'j'], waitRelease=False)
        _key_target_allKeys.extend(theseKeys)
        if len(_key_target_allKeys):
            keyTarget.keys = _key_target_allKeys[-1].name  # just the last key pressed
            keyTarget.rt = _key_target_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False

    # *gratingTarget* updates
    if gratingTarget.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        gratingTarget.frameNStart = frameN  # exact frame index
        gratingTarget.tStart = t  # local t and not account for scr refresh
        gratingTarget.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(gratingTarget, 'tStartRefresh')  # time at next scr refresh
        gratingTarget.setAutoDraw(True)

    # *textBegin* updates
    if textBegin.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        textBegin.frameNStart = frameN  # exact frame index
        textBegin.tStart = t  # local t and not account for scr refresh
        textBegin.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textBegin, 'tStartRefresh')  # time at next scr refresh
        textBegin.setAutoDraw(True)

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in targetComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "target"-------
for thisComponent in targetComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# the Routine "target" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

mocs = MocsHandler(std_ori=stim['ori'], ori_offsets=expInfo['mocs']['offsets'],
                   n_reps=expInfo['mocs']['n_reps'])
# initialise for spatial2AFC with stimuli presented simultaneously
probe_tstart = stim['tstart'][0]
foil_tstart = stim['tstart'][0]

# ------Prepare to start Routine "trial"-------
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
clockTrial.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
now = False  # used to debug issues after first trial

# ensure equal number of orders for probe and grating for each level
interval_tstarts = stim['tstart']
orders = {"interval": {}, "spatial": {}, "rotation": {}}
spatial_offset = stim['size']['deg']
for i_offset in expInfo['mocs']['offsets']:
    orders['rotation'][f"{i_offset}"] = PseudorandomOrder(items=[-1, 1], block_size=5)  # n_CW == n_CCW
    orders['interval'][f"{i_offset}"] = PseudorandomOrder(items=[[interval_tstarts[0], interval_tstarts[1]],  # n_t1st == n_t2nd
                                                                 [interval_tstarts[1], interval_tstarts[0]]],
                                                          block_size=5)
    orders['spatial'][f"{i_offset}"] = PseudorandomOrder(items=[[[-spatial_offset, 0], [spatial_offset, 0]],  # n_tleft == n_tright
                                                                [[spatial_offset, 0], [-spatial_offset, 0]]],
                                                         block_size=5)


trial_info = {'task': [], 'std': [], 'offset': [], 'rotation': [], 'first': [], 'left': [], 'correct': []
              }
# -------Run Routine "trial"-------
while mocs.continue_trials:  # continues until all levels completed for nreps
    # update trial paramaters
    level = mocs.get_trial_level()  # randomise ori_offset level
    rotation = orders['rotation'][f"{level}"].get_current_order(update=True)  # rotation in pseudorandom order
    gratingFoil.ori = stim['ori'] + (level * rotation)  # apply level with rotation (1 or -1)

    # randomise order of stimulus presentation
    if expInfo['task'] == 'temporal2AFC':
        trial_tstarts = orders['interval'][f"{level}"].get_current_order(update=True)  # defines stim order in pseudorandom manner
        probe_tstart = trial_tstarts[0]
        foil_tstart = trial_tstarts[1]
    elif expInfo['task'] == 'spatial2AFC':  # simultaneous presentation
        probe_tstart = stim['tstart'][0]
        foil_tstart = probe_tstart

    # randomise order of spatial offsets
    if expInfo['task'] == 'spatial2AFC':
        trial_positions = orders['spatial'][f"{level}"].get_current_order(update=True)
        gratingTarget.pos = trial_positions[0]
        gratingFoil.pos = trial_positions[1]
    elif expInfo['task'] == 'temporal2AFC':  # central presentation
        gratingTarget.pos = [0, 0]
        gratingFoil.pos = gratingTarget.pos

    # set correct answers dependent on interval order (t2afc) or spatial order (s2afc)
    if expInfo['task'] == 'temporal2AFC':
        trial_info['left'].append('n/a')
        if probe_tstart < foil_tstart:
            correct_key = 'f'
            trial_info['first'].append('probe')
        else:
            correct_key = 'j'
            trial_info['first'].append('foil')
    elif expInfo['task'] == 'spatial2AFC':
        trial_info['first'].append('n/a')
        if gratingTarget.pos[0] < gratingFoil.pos[0]:
            correct_key = 'f'
            trial_info['left'].append('probe')
        else:
            correct_key = 'j'
            trial_info['left'].append('foil')

    # randomise phase to minimise luminance artefacts
    for idx in range(len([gratingFoil, gratingTarget])):
        if random() > 0.5:
            [gratingFoil, gratingTarget][idx].phase = 0.25
        else:
            [gratingFoil, gratingTarget][idx].phase = -0.25

    # update trial-by-trial info (13.09.21)
    trial_info['task'].append(expInfo['task'])
    trial_info['std'].append(expInfo['orientation'])
    trial_info['offset'].append(level)
    if gratingFoil.ori == stim['ori']:
        trial_info['rotation'].append('=')
    if gratingFoil.ori < stim['ori']:
        trial_info['rotation'].append('CCW')
    else:
        trial_info['rotation'].append('CW')

    keyTrial.keys = []
    _key_trial_allKeys = []
    # update component parameters for each repeat
    tone_interval_1 = sound.Sound('620', secs=0.1, stereo=True, hamming=True)
    tone_interval_1.setVolume(.5, log=False)
    tone_interval_2 = sound.Sound('400', secs=0.1, stereo=True, hamming=True)
    tone_interval_2.setVolume(.5, log=False)
    # keep track of which components have finished
    trialComponents = [gratingTarget, gratingFoil, dotFixation, keyTrial, tone_interval_1, tone_interval_2]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    clockTrial.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    continueRoutine = True
    counter = 0
    while continueRoutine:
        # get current time
        t = clockTrial.getTime()
        tThisFlip = win.getFutureFlipTime(clock=clockTrial)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # start/stop tone_interval_1
        if tone_interval_1.status == NOT_STARTED and tThisFlip >= stim['tstart'][0] - frameTolerance:
            tone_interval_1.tStartRefresh = tThisFlipGlobal  # on global time
            tone_interval_1.play(when=win)  # sync with win flip
        if tone_interval_1.status == STARTED:
            if tThisFlipGlobal > tone_interval_1.tStartRefresh + .15 - frameTolerance:
                win.timeOnFlip(tone_interval_1, 'tStopRefresh')  # time at next scr refresh
                tone_interval_1.stop()

        # start/stop tone_interval_2 if temporal2AFC
        if expInfo['task'] == 'temporal2AFC':  # only temporal2AFC has 2 intervals
            if tone_interval_2.status == NOT_STARTED and tThisFlip >= stim['tstart'][1] - frameTolerance:
                tone_interval_2.tStartRefresh = tThisFlipGlobal  # on global time
                tone_interval_2.play(when=win)  # sync with win flip
            if tone_interval_2.status == STARTED:
                if tThisFlipGlobal > tone_interval_2.tStartRefresh + .15 - frameTolerance:
                    win.timeOnFlip(tone_interval_2, 'tStopRefresh')  # time at next scr refresh
                    tone_interval_2.stop()

        # *gratingTarget* updates
        if gratingTarget.status == NOT_STARTED and tThisFlip >= probe_tstart - frameTolerance:
            gratingTarget.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gratingTarget, 'tStartRefresh')  # time at next scr refresh
            gratingTarget.setAutoDraw(True)
        if gratingTarget.status == STARTED:
            if tThisFlipGlobal > gratingTarget.tStartRefresh + (stim["duration"] - frameTolerance):
                win.timeOnFlip(gratingTarget, 'tStopRefresh')  # time at next scr refresh
                gratingTarget.setAutoDraw(False)

        # *gratingFoil* updates
        if gratingFoil.status == NOT_STARTED and tThisFlip >= foil_tstart - frameTolerance:
            gratingFoil.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gratingFoil, 'tStartRefresh')  # time at next scr refresh
            gratingFoil.setAutoDraw(True)
        if gratingFoil.status == STARTED:
            if tThisFlipGlobal > gratingFoil.tStartRefresh + (stim["duration"] - frameTolerance):
                win.timeOnFlip(gratingFoil, 'tStopRefresh')  # time at next scr refresh
                gratingFoil.setAutoDraw(False)

        # *dotFixation* updates
        if dotFixation.status == NOT_STARTED and tThisFlip >= 0.2 - frameTolerance:
            dotFixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dotFixation, 'tStartRefresh')  # time at next scr refresh
            dotFixation.setAutoDraw(True)
        if expInfo['task'] == 'temporal2AFC':
            if dotFixation.status == STARTED and tThisFlip >= (stim['tstart'][0] - 0.1):
                dotFixation.opacity = 0  # disappear fixation dot before first stimulus
                if all(i.status == FINISHED for i in [gratingFoil, gratingTarget]):
                    dotFixation.opacity = 1  # reappear fixation dot after both stimuli

        # automate all trials on testrun
        testrun = False
        if expInfo['participant'] == 'testrun':
            testrun = True
            keyTrial.keys = 'f'
            keyTrial.corr = 1
            trial_info['correct'].append(keyTrial.corr)
            mocs.update_correct(keyTrial.corr)
            mocs.check_if_end()
            keyTrial.status = FINISHED
            continueRoutine = False
        # *keyTrial* updates
        elif keyTrial.status == NOT_STARTED and all(i.status == FINISHED for i in [gratingFoil, gratingTarget]):
            # keep track of start time/frame for later
            keyTrial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyTrial, 'tStartRefresh')  # time at next scr refresh
            keyTrial.status = STARTED
            # keyboard checking is just starting
            keyTrial.clock.reset()  # now t=0
            keyTrial.clearEvents(eventType='keyboard')
        if keyTrial.status == STARTED and not testrun:
            key_press = keyTrial.getKeys(keyList=['f', 'j'], waitRelease=False)
            if "escape" == key_press:
                endExpNow = True
            if len(key_press):
                keyTrial.keys = key_press[0].name
                if (keyTrial.keys == str(correct_key)) or (keyTrial.keys == correct_key):
                    keyTrial.corr = 1
                else:
                    keyTrial.corr = 0
                trial_info['correct'].append(keyTrial.corr)
                mocs.update_correct(keyTrial.corr)  # 1 if correct, 0 if incorrect
                mocs.check_if_end()
                # a response ends the routine
                continueRoutine = False

        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        # end routine and check ensure drawn components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            # -------Ending Routine "trial"-------
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            break
        # check if components have finished (each frame)
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

# the Routine "trial" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
thisExp.nextEntry()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

results = mocs.get_results()
thisExp.extraInfo['analysis']['results'] = results
thisExp.extraInfo['trial_info'] = pd.DataFrame(trial_info)

# these shouldn't be strictly necessary (should auto-save)
if mocs.check_if_end():
    results.to_csv(f"{filename}_results.csv")
    thisExp.saveAsPickle(filename)
    countedruns = count_runs(datadir)  # update counted runs file
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
if mocs.check_if_end():
    print(f"\n\n>>> This run on   {expInfo['task']}   for   {expInfo['orientation']}   is complete.\n")
core.quit()
