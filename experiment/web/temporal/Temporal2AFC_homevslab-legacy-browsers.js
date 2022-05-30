/******************************* 
 * Temporal2Afc_Homevslab Test *
 *******************************/

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0, 0, 0]),
  units: 'height',
  waitBlanking: true
});

// store info about the experiment session:
let expName = 'Temporal2AFC_homevslab';  // from the Builder filename that created this script
let expInfo = {'': ''};

// Start code blocks for 'Before Experiment'
// overwrite expInfo in .js file before dlg presented
expInfo = {
    "surname": "",
    "target orientation": ["vertical", "horizontal", "minus45", "plus45"],
    "monitor width (cm)": "",
    "viewing distance (cm)": ["60"],
};

function save_as_csv(rows, this_filename) {
    let csv_content = "data:text/csv;charset=utf-8,";  // put into csv format
    rows.forEach(function(rowArray) {
        let row = rowArray.join(",");
        csv_content += row + "\r\n";
    });
    // create a hidden <a> DOM node and set its download attribute as follows:
    let encodedUri = encodeURI(csv_content);
    let link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${this_filename}.csv`);
    document.body.appendChild(link);  // required for FF
    link.click();  // downloads .csv
}

var size_pix;
function deg2pix(size_deg, distance, width_cm, width_pix) {
    //deg2cm
    let visual_angle = (size_deg / 2) * (Math.PI / 180);  // 2 r.angle triangles in rad
    let adjacent = distance;  // cm
    let opposite = Math.tan(visual_angle) * adjacent;
    let size_cm = opposite * 2;  // sum opp of both triangle
    //cm2pix
    let pix_per_cm = width_pix / width_cm;
    size_pix = size_cm * pix_per_cm;
    return size_pix;
}

function randomchoice(in_array) {
    // return randomly chosen element from array
    return in_array[Math.floor(Math.random() * in_array.length)];
}
// global variables
let mocs = {"nReps":10, "nConds":7
           };

if (expInfo["surname"] == "test") {
    mocs["nReps"] = 1;
}
let exp_conditions = [];  
let thisCond = "";
let stim = {"size_deg": 8, "contrast": 0.4,
            "duration": 0.3, "interval_tstart": [0.8, 1.6]
           };

orders_set = false;
let tstarts = stim["interval_tstart"];
let Orders = {"rotation": {}, "temporal": {}};

// Class which enables pseudorandom choices from list //
class PseudorandomOrder {
    constructor(options = {}) {
        Object.assign(this, {  // assign defaults
            items: [],
            block_size: 5,
            current_order: [],
            trial_conditions: [],
            idx_keys: [],
            item_counter: {},
        }, options);
    }   
    // Method
    update_current_order() {
        // randomly choose idx_key as long as counter < block_size
        let idx_choices = []
        for (let key of this.idx_keys) {
            if (this.item_counter[key] < this.block_size) {
                idx_choices.push(key);
            }
        }
        let chosen_idx = parseInt(randomchoice(idx_choices))
        this.current_order = this.items[chosen_idx]
        this.item_counter[chosen_idx] = this.item_counter[chosen_idx] + 1
        // if all reached block_size then reset counters
        let fin = []
        for (let i of this.idx_keys) {
            fin.push(this.item_counter[i] == this.block_size);
        }
        function finished(el) {
            return el;
        }
        if (fin.every(finished) == true) {
            for (let i in this.idx_keys) {
                this.item_counter[i] = 0;
            }
        }
        return this.current_order
    }
    // Methods
    initialise() {
        // set vals of idx_keys
        for (let idx in this.items) {
            this.idx_keys.push(idx.toString());
        }
        // set counter to zero for each order
        for (let key of this.idx_keys) {
            this.item_counter[key] = 0;
        }
    }
}
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(custom_codeRoutineBegin());
flowScheduler.add(custom_codeRoutineEachFrame());
flowScheduler.add(custom_codeRoutineEnd());
flowScheduler.add(instructionsRoutineBegin());
flowScheduler.add(instructionsRoutineEachFrame());
flowScheduler.add(instructionsRoutineEnd());
flowScheduler.add(targetRoutineBegin());
flowScheduler.add(targetRoutineEachFrame());
flowScheduler.add(targetRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin, trialsLoopScheduler);
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);
flowScheduler.add(endscreenRoutineBegin());
flowScheduler.add(endscreenRoutineEachFrame());
flowScheduler.add(endscreenRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    {'name': 'resources/PCN_minus45_12.50deg.png', 'path': 'resources/PCN_minus45_12.50deg.png'},
    {'name': 'resources/PCN_horizontal_2.50deg.png', 'path': 'resources/PCN_horizontal_2.50deg.png'},
    {'name': 'resources/PCN_minus45_17.86deg.png', 'path': 'resources/PCN_minus45_17.86deg.png'},
    {'name': 'resources/PCN_vertical_4.29deg.png', 'path': 'resources/PCN_vertical_4.29deg.png'},
    {'name': 'resources/PCN_horizontal_0.36deg.png', 'path': 'resources/PCN_horizontal_0.36deg.png'},
    {'name': 'resources/PCN_plus45_12.50deg.png', 'path': 'resources/PCN_plus45_12.50deg.png'},
    {'name': 'resources/PCN_vertical_0.36deg.png', 'path': 'resources/PCN_vertical_0.36deg.png'},
    {'name': 'resources/PCN_horizontal_3.57deg.png', 'path': 'resources/PCN_horizontal_3.57deg.png'},
    {'name': 'resources/PCN_horizontal_1.79deg.png', 'path': 'resources/PCN_horizontal_1.79deg.png'},
    {'name': 'resources/PCN_plus45_17.86deg.png', 'path': 'resources/PCN_plus45_17.86deg.png'},
    {'name': 'resources/PCN_horizontal_0.00deg.png', 'path': 'resources/PCN_horizontal_0.00deg.png'},
    {'name': 'resources/PCN_plus45_0.00deg.png', 'path': 'resources/PCN_plus45_0.00deg.png'},
    {'name': 'resources/PCN_minus45_7.14deg.png', 'path': 'resources/PCN_minus45_7.14deg.png'},
    {'name': 'resources/PCN_minus45_1.79deg.png', 'path': 'resources/PCN_minus45_1.79deg.png'},
    {'name': 'resources/PCN_plus45_1.79deg.png', 'path': 'resources/PCN_plus45_1.79deg.png'},
    {'name': 'resources/PCN_vertical_1.07deg.png', 'path': 'resources/PCN_vertical_1.07deg.png'},
    {'name': 'resources/PCN_vertical_0.00deg.png', 'path': 'resources/PCN_vertical_0.00deg.png'},
    {'name': 'resources/PCN_vertical_2.86deg.png', 'path': 'resources/PCN_vertical_2.86deg.png'},
    {'name': 'resources/PCN_minus45_3.57deg.png', 'path': 'resources/PCN_minus45_3.57deg.png'},
    {'name': 'resources/PCN_horizontal_4.29deg.png', 'path': 'resources/PCN_horizontal_4.29deg.png'},
    {'name': 'resources/PCN_vertical_3.57deg.png', 'path': 'resources/PCN_vertical_3.57deg.png'},
    {'name': 'resources/PCN_vertical_5.00deg.png', 'path': 'resources/PCN_vertical_5.00deg.png'},
    {'name': 'resources/PCN_plus45_7.14deg.png', 'path': 'resources/PCN_plus45_7.14deg.png'},
    {'name': 'resources/PCN_plus45_10.71deg.png', 'path': 'resources/PCN_plus45_10.71deg.png'},
    {'name': 'resources/PCN_minus45_0.00deg.png', 'path': 'resources/PCN_minus45_0.00deg.png'},
    {'name': 'resources/PCN_vertical_2.50deg.png', 'path': 'resources/PCN_vertical_2.50deg.png'},
    {'name': 'resources/PCN_horizontal_1.43deg.png', 'path': 'resources/PCN_horizontal_1.43deg.png'},
    {'name': 'resources/PCN_minus45_25.00deg.png', 'path': 'resources/PCN_minus45_25.00deg.png'},
    {'name': 'resources/PCN_vertical_0.71deg.png', 'path': 'resources/PCN_vertical_0.71deg.png'},
    {'name': 'resources/PCN_plus45_8.93deg.png', 'path': 'resources/PCN_plus45_8.93deg.png'},
    {'name': 'resources/PCN_plus45_5.36deg.png', 'path': 'resources/PCN_plus45_5.36deg.png'},
    {'name': 'resources/PCN_plus45_25.00deg.png', 'path': 'resources/PCN_plus45_25.00deg.png'},
    {'name': 'resources/PCN_vertical_1.43deg.png', 'path': 'resources/PCN_vertical_1.43deg.png'},
    {'name': 'resources/PCN_plus45_21.43deg.png', 'path': 'resources/PCN_plus45_21.43deg.png'},
    {'name': 'resources/PCN_vertical_2.14deg.png', 'path': 'resources/PCN_vertical_2.14deg.png'},
    {'name': 'resources/PCN_horizontal_5.00deg.png', 'path': 'resources/PCN_horizontal_5.00deg.png'},
    {'name': 'resources/PCN_horizontal_1.07deg.png', 'path': 'resources/PCN_horizontal_1.07deg.png'},
    {'name': 'resources/PCN_plus45_14.29deg.png', 'path': 'resources/PCN_plus45_14.29deg.png'},
    {'name': 'resources/PCN_minus45_5.36deg.png', 'path': 'resources/PCN_minus45_5.36deg.png'},
    {'name': 'resources/PCN_vertical_1.79deg.png', 'path': 'resources/PCN_vertical_1.79deg.png'},
    {'name': 'resources/PCN_minus45_21.43deg.png', 'path': 'resources/PCN_minus45_21.43deg.png'},
    {'name': 'resources/PCN_minus45_8.93deg.png', 'path': 'resources/PCN_minus45_8.93deg.png'},
    {'name': 'resources/PCN_minus45_14.29deg.png', 'path': 'resources/PCN_minus45_14.29deg.png'},
    {'name': 'resources/PCN_minus45_10.71deg.png', 'path': 'resources/PCN_minus45_10.71deg.png'},
    {'name': 'resources/PCN_plus45_3.57deg.png', 'path': 'resources/PCN_plus45_3.57deg.png'},
    {'name': 'resources/PCN_horizontal_2.14deg.png', 'path': 'resources/PCN_horizontal_2.14deg.png'},
    {'name': 'resources/PCN_horizontal_0.71deg.png', 'path': 'resources/PCN_horizontal_0.71deg.png'},
    {'name': 'resources/PCN_horizontal_2.86deg.png', 'path': 'resources/PCN_horizontal_2.86deg.png'}
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.DEBUG);


var frameDur;
function updateInfo() {
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2020.2.8';
  expInfo['OS'] = window.navigator.platform;

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  psychoJS.setRedirectUrls('https://run.pavlovia.org/RichardL/temporal2afc_homevslab', 'https://run.pavlovia.org/RichardL/temporal2afc_homevslab');

  return Scheduler.Event.NEXT;
}


var custom_codeClock;
var filename;
var instructionsClock;
var text_instructions;
var key_instructions;
var targetClock;
var text_target;
var key_target;
var image_target;
var text_begin;
var trialClock;
var tone_interval1;
var tone_interval2;
var grating_target;
var grating_foil;
var dot_fixation;
var dot_fixation2;
var key_trial;
var endscreenClock;
var text_end;
var summary_data;
var globalClock;
var routineTimer;
function experimentInit() {
  // Initialize components for Routine "custom_code"
  custom_codeClock = new util.Clock();
  // end experiment if any dlg fields left blank
  for (var key in expInfo) {
      if (expInfo[key] === "") {  //value not entered
  //        quitPsychoJS("ENTER ALL USER INFO", false");
              psychoJS.event.quit;
      }
  }
  filename = `${expInfo["expName"]}_${expInfo["surname"]}_${expInfo["target orientation"]}_${expInfo["date"]}`;
  // Initialize components for Routine "instructions"
  instructionsClock = new util.Clock();
  text_instructions = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_instructions',
    text: 'Two gratings will be presented one after the other in the centre of your screen.\n\nRespond with the ‘F’ key if the target orientation appears 1st.\n\nRespond with the ‘J’ key if the target orientation appears 2nd.\n\nIf you are unsure then respond with your best guess.\n\n\n\nTo continue and view your target orientation, press ‘F’ or ‘J’.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], height: 0.075,  wrapWidth: 2, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_instructions = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "target"
  targetClock = new util.Clock();
  text_target = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_target',
    text: 'This is your target grating.\n\nJudge which of two gratings is at this orientation.\n\npress ‘F’ if 1st\npress ‘J’ if 2nd',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.7], height: 0.075,  wrapWidth: 2, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_target = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  image_target = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_target', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0, 0],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  text_begin = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_begin',
    text: "press 'F' or 'J' to begin",
    font: 'Arial',
    units: 'norm', 
    pos: [0, (- 0.7)], height: 0.075,  wrapWidth: 2, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: -3.0 
  });
  
  // initialise val to global
  size_pix = 0;
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  tone_interval1 = new sound.Sound({
    win: psychoJS.window,
    value: '640',
    secs: 0.15,
    });
  tone_interval1.setVolume(0.2);
  tone_interval2 = new sound.Sound({
    win: psychoJS.window,
    value: '420',
    secs: 0.15,
    });
  tone_interval2.setVolume(0.2);
  grating_target = new visual.ImageStim({
    win : psychoJS.window,
    name : 'grating_target', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0, 0],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : true, flipVert : true,
    texRes : 512, interpolate : true, depth : -2.0 
  });
  grating_foil = new visual.ImageStim({
    win : psychoJS.window,
    name : 'grating_foil', units : 'pix', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0, 0],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 512, interpolate : true, depth : -3.0 
  });
  dot_fixation = new visual.Polygon ({
    win: psychoJS.window, name: 'dot_fixation', units : 'pix', 
    edges: 10, size:[0, 0],
    ori: 0, pos: [0, 0],
    lineWidth: 0, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1.0), (- 1.0), (- 1.0)]),
    opacity: 1, depth: -4, interpolate: true,
  });
  
  dot_fixation2 = new visual.Polygon ({
    win: psychoJS.window, name: 'dot_fixation2', units : 'pix', 
    edges: 10, size:[0, 0],
    ori: 0, pos: [0, 0],
    lineWidth: 0, lineColor: new util.Color([(- 1), (- 1), (- 1)]),
    fillColor: new util.Color([(- 1.0), (- 1.0), (- 1.0)]),
    opacity: 1, depth: -5, interpolate: true,
  });
  
  key_trial = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // test
  if (expInfo["surname"] == "test") {
      mocs["nReps"] = 1;
  }
  
  // different ranges for cardinal and oblique stimuli
  if ((expInfo["target orientation"] === "vertical")
     || (expInfo["target orientation"] === "horizontal")) {
      expInfo["cardinal"] = true;
      if (expInfo["surname"] == "Leadbeater") {
          mocs["offsets"] = ["0.36", "0.71", "1.07", "1.43", "1.79", "2.14", "2.50"];
      } else { 
          mocs["offsets"] = ["0.71", "1.43", "2.14", "2.86", "3.57", "4.29", "5.00"];
      }
  } else {
      expInfo["cardinal"] = false;
      if (expInfo["surname"] == "Leadbeater") {
          mocs["offsets"] = ["1.79", "3.57", "5.36", "7.14", "8.93", "10.71", "12.50"];
      } else { 
          mocs["offsets"] = ["3.57", "7.14", "10.71", "14.29", "17.86", "21.43", "25.00"];
      }
  }
  
  // Class declaration
  class ExperimentCondition {
      constructor(options = {}) {
          Object.assign(this, {  // assign defaults
              orientation: "vertical",
              ori_diff: "0.00",
              current_count: 0,
              max_count: 10,
              nCorrect: 0,
              }, options);
      }
      // Getter
      get trial_image() {
          if (this.current_count < this.max_count) {
              this.current_count += 1;   // add to counter when getter is used
              return this.retrieve_filename();
          } else {
              return "";  // do not return image if max_count reached
          }
      }
      get proportion_correct() {
          return this.calc_proportion_corr();
      }
      // Method
      retrieve_filename() {
          return `resources/PCN_${this.orientation}_${this.ori_diff}deg.png`;
      }
      calc_proportion_corr() {
          if (this.current_count > 0) {
              return (1 / this.current_count) * this.nCorrect;
          } else {
              return "";
          }
      }
  }
  
  // instance of ExperimentCondition for each cond
  for (let offset of mocs["offsets"]) {
      let i_cond = new ExperimentCondition({orientation: expInfo["target orientation"], 
                                            ori_diff: offset, 
                                            max_count:mocs["nReps"]});
      exp_conditions.push(i_cond);
      }
  //// Constant Stim Properties ////
  // Image //
  grating_target.image = `resources/PCN_${expInfo['target orientation']}_${"0.00"}deg.png`;
  
  // Class which enables pseudorandom choices from list //
  class PseudorandomOrder {
      constructor(options = {}) {
          Object.assign(this, {  // assign defaults
              items: [],
              block_size: 5,
              current_order: [],
              trial_conditions: [],
              idx_keys: [],
              item_counter: {},
          }, options);
      }   
      // Method
      update_current_order() {
          // randomly choose idx_key as long as counter < block_size
          let idx_choices = []
          for (let key of this.idx_keys) {
              if (this.item_counter[key] < this.block_size) {
                  idx_choices.push(key);
              }
          }
          let chosen_idx = parseInt(randomchoice(idx_choices))
          this.current_order = this.items[chosen_idx]
          this.item_counter[chosen_idx] = this.item_counter[chosen_idx] + 1
          // if all reached block_size then reset counters
          let fin = []
          for (let i of this.idx_keys) {
              fin.push(this.item_counter[i] == this.block_size);
          }
          function finished(el) {
              return el;
          }
          if (fin.every(finished) == true) {
              for (let i in this.idx_keys) {
                  this.item_counter[i] = 0;
              }
          }
          return this.current_order
      }
      // Methods
      initialise() {
          // set vals of idx_keys
          for (let idx in this.items) {
              this.idx_keys.push(idx.toString());
          }
          // set counter to zero for each order
          for (let key of this.idx_keys) {
              this.item_counter[key] = 0;
          }
      }
  }
  // Initialize components for Routine "endscreen"
  endscreenClock = new util.Clock();
  text_end = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_end',
    text: 'END OF RUN!\n\n…SAVING DATA…',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], height: 0.1,  wrapWidth: 1.3, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  summary_data = [["Experiment Name", "Target Orientation", "Orientation Difference (deg)", "Proportion Correct"]
  ];
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var custom_codeComponents;
function custom_codeRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'custom_code'-------
    t = 0;
    custom_codeClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    expInfo["completed"] = false;
    // keep track of which components have finished
    custom_codeComponents = [];
    
    custom_codeComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


var continueRoutine;
function custom_codeRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'custom_code'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = custom_codeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    custom_codeComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function custom_codeRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'custom_code'-------
    custom_codeComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "custom_code" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_instructions_allKeys;
var instructionsComponents;
function instructionsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instructions'-------
    t = 0;
    instructionsClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_instructions.keys = undefined;
    key_instructions.rt = undefined;
    _key_instructions_allKeys = [];
    // keep track of which components have finished
    instructionsComponents = [];
    instructionsComponents.push(text_instructions);
    instructionsComponents.push(key_instructions);
    
    instructionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instructionsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instructions'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_instructions* updates
    if (t >= 0.0 && text_instructions.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_instructions.tStart = t;  // (not accounting for frame time here)
      text_instructions.frameNStart = frameN;  // exact frame index
      
      text_instructions.setAutoDraw(true);
    }

    
    // *key_instructions* updates
    if (t >= 0.0 && key_instructions.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_instructions.tStart = t;  // (not accounting for frame time here)
      key_instructions.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_instructions.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_instructions.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_instructions.clearEvents(); });
    }

    if (key_instructions.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_instructions.getKeys({keyList: ['y', 'n', 'left', 'right', 'space', 'f', 'j'], waitRelease: false});
      _key_instructions_allKeys = _key_instructions_allKeys.concat(theseKeys);
      if (_key_instructions_allKeys.length > 0) {
        key_instructions.keys = _key_instructions_allKeys[_key_instructions_allKeys.length - 1].name;  // just the last key pressed
        key_instructions.rt = _key_instructions_allKeys[_key_instructions_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instructionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instructionsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instructions'-------
    instructionsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_target_allKeys;
var targetComponents;
function targetRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'target'-------
    t = 0;
    targetClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_target.keys = undefined;
    key_target.rt = undefined;
    _key_target_allKeys = [];
    // Size //
    size_pix = deg2pix(stim["size_deg"], expInfo["viewing distance (cm)"],
                       expInfo["monitor width (cm)"], psychoJS.window.size[0]
    );  // convert size to pixels using monitor info
    // Contrast //
    grating_target.opacity = stim["contrast"];
    
    // Image //
    image_target.image = `resources/PCN_${expInfo['target orientation']}_${"0.00"}deg.png`;
    
    // Size //
    image_target.size = [size_pix, size_pix];
    
    // keep track of which components have finished
    targetComponents = [];
    targetComponents.push(text_target);
    targetComponents.push(key_target);
    targetComponents.push(image_target);
    targetComponents.push(text_begin);
    
    targetComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function targetRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'target'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = targetClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_target* updates
    if (t >= 0.0 && text_target.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_target.tStart = t;  // (not accounting for frame time here)
      text_target.frameNStart = frameN;  // exact frame index
      
      text_target.setAutoDraw(true);
    }

    
    // *key_target* updates
    if (t >= 0.0 && key_target.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_target.tStart = t;  // (not accounting for frame time here)
      key_target.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_target.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_target.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_target.clearEvents(); });
    }

    if (key_target.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_target.getKeys({keyList: ['y', 'n', 'left', 'right', 'space', 'f', 'j'], waitRelease: false});
      _key_target_allKeys = _key_target_allKeys.concat(theseKeys);
      if (_key_target_allKeys.length > 0) {
        key_target.keys = _key_target_allKeys[_key_target_allKeys.length - 1].name;  // just the last key pressed
        key_target.rt = _key_target_allKeys[_key_target_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *image_target* updates
    if (t >= 0.0 && image_target.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_target.tStart = t;  // (not accounting for frame time here)
      image_target.frameNStart = frameN;  // exact frame index
      
      image_target.setAutoDraw(true);
    }

    
    // *text_begin* updates
    if (t >= 0.0 && text_begin.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_begin.tStart = t;  // (not accounting for frame time here)
      text_begin.frameNStart = frameN;  // exact frame index
      
      text_begin.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    targetComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function targetRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'target'-------
    targetComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "target" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var trials;
var currentLoop;
function trialsLoopBegin(trialsLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials = new TrialHandler({
    psychoJS: psychoJS,
    nReps: (mocs["nReps"] * mocs["nConds"]), method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'trials'
  });
  psychoJS.experiment.addLoop(trials); // add the loop to the experiment
  currentLoop = trials;  // we're now the current loop

  // Schedule all the trials in the trialList:
  trials.forEach(function() {
    const snapshot = trials.getSnapshot();

    trialsLoopScheduler.add(importConditions(snapshot));
    trialsLoopScheduler.add(trialRoutineBegin(snapshot));
    trialsLoopScheduler.add(trialRoutineEachFrame(snapshot));
    trialsLoopScheduler.add(trialRoutineEnd(snapshot));
    trialsLoopScheduler.add(endLoopIteration(trialsLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function trialsLoopEnd() {
  psychoJS.experiment.removeLoop(trials);

  return Scheduler.Event.NEXT;
}


var _key_trial_allKeys;
var orders_set;
var dot_tstart;
var target_t;
var foil_t;
var trialComponents;
function trialRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'trial'-------
    t = 0;
    trialClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    tone_interval1.secs=0.15;
    tone_interval1.setVolume(0.2);
    tone_interval2.secs=0.15;
    tone_interval2.setVolume(0.2);
    key_trial.keys = undefined;
    key_trial.rt = undefined;
    _key_trial_allKeys = [];
    // only use instances where .current_count() < params["nReps"]
    let trial_conditions = [];
    for (let i_cond of exp_conditions) {
        if (i_cond.current_count < mocs["nReps"]) {
            trial_conditions.push(i_cond);
            }
        }
    // if all conds have reached nReps then end experiment
    if (trial_conditions == "") {
        console.log("end of experiment")  // todo force end of exp
    } else {
        // randomise which condition to use
        let cond_idx = Math.floor(Math.random() * trial_conditions.length);
        thisCond = trial_conditions[cond_idx];  // global
        grating_foil.image = thisCond.trial_image;
    }
    //// ORDERS ////
    
    if (orders_set == false) {  // only runs on first trial
        for (let i_offset of mocs["offsets"]) {
            // Determinine spatial offsets in pseudorandom fashion //
            Orders["temporal"][i_offset] = new PseudorandomOrder({items: [[tstarts[0], tstarts[1]],
                                                                        [tstarts[1], tstarts[0]]],
                                                                block_size: 5}
                                                            );
            Orders["temporal"][i_offset].initialise()
    
            // Determine CW or CCW in pseudorandom fashion //
            Orders["rotation"][i_offset] = new PseudorandomOrder({items: [0, 2], 
                                                                 block_size: 5}
                                                             );
            Orders["rotation"][i_offset].initialise()
        }
        orders_set = true;  // code wont rerun
    }
    
    
    //// FIXED PROPERTIES ////
    
    // Size //
    grating_target.size = [size_pix, size_pix];
    grating_foil.size = [size_pix, size_pix];
    dot_fixation.size = [(size_pix/70), (size_pix/70)];
    dot_tstart = 0.3;
    
    // Contrast //
    image_target.opacity = stim["contrast"];
    grating_foil.opacity = stim["contrast"];
    dot_fixation.opacity = (stim["contrast"]/1.4);
    
    
    //// FLEXIBLE PROPERTIES ////
    
    // Rotation //
    let trial_rotation = Orders["rotation"][thisCond.ori_diff].update_current_order()
    grating_foil.ori = thisCond.ori_diff * trial_rotation  // *0=CW, *1=target, *2=CCW 
    if (trial_rotation == 0) {
        trials.extraInfo["rotation"] = "CW";
    } else {
        trials.extraInfo["rotation"] = "CCW";
    }  // mirror images cause no issues with pixel scaling
    
    
    // Interval Order //
    let trial_temporal_order = Orders["temporal"][thisCond.ori_diff].update_current_order()
    target_t = trial_temporal_order[0];
    foil_t = trial_temporal_order[1];
    if (target_t < foil_t) {
        trials.extraInfo["corrAns"] = "f";
    } else {
        trials.extraInfo["corrAns"] = "j";
    }
    
    
    // Phase //
    if (Math.random() >= 0.5) {
        grating_target.flipVert = true;
        grating_target.flipHoriz = true;
    } else {  // flipping both ways creates a mirror image
        grating_target.flipVert = false;
        grating_target.flipHoriz = false;
    }
    if (Math.random() >= 0.5) {
        grating_foil.flipVert = true;
        grating_foil.flipHoriz = true;
    } else {
        grating_foil.flipVert = false;
        grating_foil.flipHoriz = false;
    }
    if (expInfo["surname"] == "testrun") {
        // automate responses to end exp quickly with full n_reps
        foil_t = 0;
        target_t = 0;
        stim["duration"] = 0.01;
        stim["interval_tstart"] = [0, 0];
        dot_tstart = 0;
        key_trial.corr = 1;
       
        // give vals to allow later code to run
        key_trial.status = PsychoJS.Status.FINISHED;
        key_trial.tStart = 0;
        key_trial.rt = 0;
        key_trial.keys = 'f'; 
    }
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(tone_interval1);
    trialComponents.push(tone_interval2);
    trialComponents.push(grating_target);
    trialComponents.push(grating_foil);
    trialComponents.push(dot_fixation);
    trialComponents.push(dot_fixation2);
    trialComponents.push(key_trial);
    
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


var frameRemains;
function trialRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'trial'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // start/stop tone_interval1
    if (t >= stim["interval_tstart"][0] && tone_interval1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      tone_interval1.tStart = t;  // (not accounting for frame time here)
      tone_interval1.frameNStart = frameN;  // exact frame index
      
      psychoJS.window.callOnFlip(function(){ tone_interval1.play(); });  // screen flip
      tone_interval1.status = PsychoJS.Status.STARTED;
    }
    frameRemains = stim["interval_tstart"][0] + 0.15 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((tone_interval1.status === PsychoJS.Status.STARTED || tone_interval1.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      if (0.15 > 0.5) {  tone_interval1.stop();  // stop the sound (if longer than duration)
        tone_interval1.status = PsychoJS.Status.FINISHED;
      }
    }
    // start/stop tone_interval2
    if (t >= stim["interval_tstart"][1] && tone_interval2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      tone_interval2.tStart = t;  // (not accounting for frame time here)
      tone_interval2.frameNStart = frameN;  // exact frame index
      
      tone_interval2.play();  // start the sound (it finishes automatically)
      tone_interval2.status = PsychoJS.Status.STARTED;
    }
    frameRemains = stim["interval_tstart"][1] + 0.15 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((tone_interval2.status === PsychoJS.Status.STARTED || tone_interval2.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      if (0.15 > 0.5) {  tone_interval2.stop();  // stop the sound (if longer than duration)
        tone_interval2.status = PsychoJS.Status.FINISHED;
      }
    }
    
    // *grating_target* updates
    if (t >= target_t && grating_target.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      grating_target.tStart = t;  // (not accounting for frame time here)
      grating_target.frameNStart = frameN;  // exact frame index
      
      grating_target.setAutoDraw(true);
    }

    frameRemains = target_t + stim["duration"] - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((grating_target.status === PsychoJS.Status.STARTED || grating_target.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      grating_target.setAutoDraw(false);
    }
    
    // *grating_foil* updates
    if (t >= foil_t && grating_foil.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      grating_foil.tStart = t;  // (not accounting for frame time here)
      grating_foil.frameNStart = frameN;  // exact frame index
      
      grating_foil.setAutoDraw(true);
    }

    frameRemains = foil_t + stim["duration"] - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((grating_foil.status === PsychoJS.Status.STARTED || grating_foil.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      grating_foil.setAutoDraw(false);
    }
    
    // *dot_fixation* updates
    if (t >= 0 && dot_fixation.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      dot_fixation.tStart = t;  // (not accounting for frame time here)
      dot_fixation.frameNStart = frameN;  // exact frame index
      
      dot_fixation.setAutoDraw(true);
    }

    frameRemains = 0.7  - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (dot_fixation.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      dot_fixation.setAutoDraw(false);
    }
    
    // *dot_fixation2* updates
    if (t >= ((stim["interval_tstart"][1] + stim["duration"]) + 0.1) && dot_fixation2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      dot_fixation2.tStart = t;  // (not accounting for frame time here)
      dot_fixation2.frameNStart = frameN;  // exact frame index
      
      dot_fixation2.setAutoDraw(true);
    }

    
    // *key_trial* updates
    if (t >= (stim["interval_tstart"][1] + stim["duration"]) && key_trial.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_trial.tStart = t;  // (not accounting for frame time here)
      key_trial.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      key_trial.clock.reset();
      key_trial.start();
      key_trial.clearEvents();
    }

    if (key_trial.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_trial.getKeys({keyList: ['f', 'j'], waitRelease: false});
      _key_trial_allKeys = _key_trial_allKeys.concat(theseKeys);
      if (_key_trial_allKeys.length > 0) {
        key_trial.keys = _key_trial_allKeys[0].name;  // just the first key pressed
        key_trial.rt = _key_trial_allKeys[0].rt;
        // was this correct?
        if (key_trial.keys == trials.extraInfo["corrAns"]) {
            key_trial.corr = 1;
        } else {
            key_trial.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    if (expInfo["surname"] == "testrun" && continueRoutine == true) {
        continueRoutine = false;
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function trialRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'trial'-------
    trialComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    tone_interval1.stop();  // ensure sound has stopped at end of routine
    tone_interval2.stop();  // ensure sound has stopped at end of routine
    // was no response the correct answer?!
    if (key_trial.keys === undefined) {
      if (['None','none',undefined].includes(trials.extraInfo["corrAns"])) {
         key_trial.corr = 1;  // correct non-response
      } else {
         key_trial.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for thisExp (ExperimentHandler)
    psychoJS.experiment.addData('key_trial.keys', key_trial.keys);
    psychoJS.experiment.addData('key_trial.corr', key_trial.corr);
    if (typeof key_trial.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_trial.rt', key_trial.rt);
        routineTimer.reset();
        }
    
    key_trial.stop();
    thisCond.nCorrect += key_trial.corr;  // +1 if corr else +0
    trials.extraInfo["nCorrect"] = thisCond.nCorrect;
    trials.extraInfo["ori offset"] = thisCond.ori_diff;
    trials.extraInfo["current count"] = thisCond.current_count;
    trials.extraInfo["proportion correct"] = thisCond.proportion_correct;
    // the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var endscreenComponents;
function endscreenRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'endscreen'-------
    t = 0;
    endscreenClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    // keep track of which components have finished
    endscreenComponents = [];
    endscreenComponents.push(text_end);
    
    endscreenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function endscreenRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'endscreen'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = endscreenClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_end* updates
    if (t >= 0.0 && text_end.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_end.tStart = t;  // (not accounting for frame time here)
      text_end.frameNStart = frameN;  // exact frame index
      
      text_end.setAutoDraw(true);
    }

    frameRemains = 0.0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((text_end.status === PsychoJS.Status.STARTED || text_end.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      text_end.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    endscreenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function endscreenRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'endscreen'-------
    endscreenComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    expInfo["completed"] = true;    
    return Scheduler.Event.NEXT;
  };
}


function endLoopIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        const thisTrial = snapshot.getCurrentTrial();
        if (typeof thisTrial === 'undefined' || !('isTrials' in thisTrial) || thisTrial.isTrials) {
          psychoJS.experiment.nextEntry(snapshot);
        }
      }
    return Scheduler.Event.NEXT;
    }
  };
}


function importConditions(currentLoop) {
  return function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  ////summary_data = [["Experiment Name", "Target Orientation", "Orientation Difference (deg)", "Proportion Correct"]];
  //if (expInfo["completed"] == true) {
  //    for (let i_cond of exp_conditions) {
  //        summary_data.push([expInfo["expName"], expInfo["target orientation"],
  //                           i_cond.ori_diff, i_cond.proportion_correct
  //        ])
  //    }
  //    save_as_csv(summary_data, `myData_${filename}`);
  //}
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
