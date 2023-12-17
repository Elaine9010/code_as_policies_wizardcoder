## this script evaluates the performance of a wizardcoder agent with self-defined tasks
## the questions are split into 3 cases: simple single-step tasks, complex single-step tasks, and multi-step tasks
## the success rate of each case is saved in the file

model_size = "7B"
save_flag = True
append_failure = False

max_tokens = 256,
temperature = 0.5
prompt_type = "prompt0"


from copy import copy
from tqdm.auto import trange
from time import sleep

import os
import ast
import astunparse
import numpy as np

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from vllm import LLM, SamplingParams #wizardCoder
from datetime import datetime

#save evalutation results to a file

file_name = "eval_score_"+model_size+prompt_type+"_temp_"+str(temperature) + ".txt"
file_path = os.path.join(os.getcwd(), "Eval_results_diff_prompts_wizardcoder", file_name)

def save_to_file(file_path, result):
    file = open(file_path, "a")
    file.write(result)
    file.write("\n")
    file.close()

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
message = "--------------------------------" + current_time + "--------------------------------------\n"
save_to_file(file_path, message)


def exec_safe(code_str, gvars=None, lvars=None):
  banned_phrases = ['import', '__']
  #for phrase in banned_phrases:
  #  assert phrase not in code_str

  if gvars is None:
    gvars = {}
  if lvars is None:
    lvars = {}
  
  empty_fn = lambda *args, **kwargs: None
  custom_gvars = merge_dicts([
      gvars,
      {'exec': empty_fn, 'eval': empty_fn}
  ])

  try:
    exec(code_str, custom_gvars, lvars)
  except Exception as e:
    print(f"An error occurred: {e}")
    lvars['ret_val'] = None


#load wizardcoder model
base_model="WizardLM/WizardCoder-Python-{f}-V1.0".format(f=model_size)
llm = LLM(model=base_model, tensor_parallel_size=1)

def lmp_batch(base_prompt, cmds, stop_tokens=None, strip=False, batch_size=20, rate_limit_time=5, query_kwargs=None):
    prompts = [
      f'{base_prompt}\n{cmd}'
      for cmd in cmds
    ]

    use_query_kwargs = copy(default_query_kwargs)
    if query_kwargs is not None:
      use_query_kwargs.update(query_kwargs)

    responses = []
    for start_idx in trange(0, len(prompts), batch_size):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx : end_idx]
        #raw_responses_batch = openai.Completion.create(prompt=batch_prompts, stop=stop_tokens, **use_query_kwargs)
        print("-----------------------------------------------------------------------------------------------------------------------------")
        print("prompting wizardcoder with temperature: ", use_query_kwargs['temperature'], " and max_tokens: ", use_query_kwargs['max_tokens'])
        sampling_params = SamplingParams(temperature=use_query_kwargs['temperature'], top_p=1, max_tokens=use_query_kwargs['max_tokens'], stop=stop_tokens)
        raw_responses_batch = llm.generate(batch_prompts, sampling_params)
        
        responses_batch = [
            r.outputs[0].text
            for r in raw_responses_batch
        ]
        if strip:
            responses_batch = [response.strip() for response in responses_batch]
        responses.extend(responses_batch)

        if end_idx != len(prompts):
            sleep(rate_limit_time)

    return responses

def lmp(base_prompt, query, stop_tokens=None, log=True, return_response=False, query_kwargs=None):
    new_prompt = f'{base_prompt}\n{query}'
    use_query_kwargs = copy(default_query_kwargs)
    if query_kwargs is not None:
      use_query_kwargs.update(query_kwargs)
    #response = openai.Completion.create(prompt=new_prompt, stop=stop_tokens, **use_query_kwargs)['choices'][0]['text'].strip()
    print("-----------------------------------------------------------------------------------------------------------------------------")
    print("prompting wizardcoder with temperature: ", use_query_kwargs['temperature'], " and max_tokens: ", use_query_kwargs['max_tokens'])
    sampling_params = SamplingParams(temperature=use_query_kwargs['temperature'], top_p=1, max_tokens=use_query_kwargs['max_tokens'], stop=stop_tokens)
    response = llm.generate(new_prompt, sampling_params)[0].outputs[0].text

    if log:
      print(query)
      print(response)

    if return_response:
      return response

def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }

def create_new_fs_from_code(prompt_f_gen, code_str, context_vars, return_src=False):
    fs, f_assigns = {}, {}
    f_parser = FunctionParser(fs, f_assigns)
    f_parser.visit(ast.parse(code_str))
    for f_name, f_assign in f_assigns.items():
        if f_name in fs:
            fs[f_name] = f_assign

    new_fs = {}
    srcs = {}
    for f_name, f_sig in fs.items():
        all_vars = merge_dicts([context_vars, new_fs])
        if not var_exists(f_name, all_vars):
            f, f_src = lmp_fgen(prompt_f_gen, f_name, f_sig, recurse=True, context_vars=all_vars, return_src=True)

            new_fs[f_name], srcs[f_name] = f, f_src

    if return_src:
        return new_fs, srcs
    return new_fs

def lmp_fgen(prompt, f_name, f_sig, stop_tokens=['# define function:', '# example:'], recurse=False, 
             context_vars=None, bug_fix=False, log=True, return_src=False, query_kwargs=None, info=''):
    query = f'# define function: {f_sig}.'
    if info:
      query = f'{query}\n# info: {info}.'
    f_src = lmp(prompt, query, stop_tokens=stop_tokens, log=False, return_response=True, query_kwargs=query_kwargs)
    
    if context_vars is None:
        context_vars = {}
    gvars = context_vars
    lvars = {}

    f_success = True
    try:
      exec_safe(f_src, gvars, lvars)
      f = lvars[f_name]
    except Exception as e:
      print(e)
      f = lambda *args, **kargs: None
      f_success = False 

    if recurse and f_success:
      f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
      child_fs, child_f_srcs = create_new_fs_from_code(prompt_f_gen, f_def_body, context_vars, return_src=True)
      
      if len(child_fs) > 0:
        # redefine parent f so newly created child_fs are in scope
        gvars = merge_dicts([context_vars, child_fs])
        lvars = {}
      
        exec_safe(f_src, gvars, lvars)
        
        f = lvars[f_name]

    if log:
        to_print = highlight(f'{query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')

    if return_src:
        return f, f_src
    return f

class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
      super().__init__()
      self._fs = fs
      self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node

def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists

def eval_questions_lmp(prompt_f_gen, prompt, questions, answers, context_vars):
  queries = questions
  resps = lmp_batch(
      prompt,
      queries,
      strip=True,
      stop_tokens=['objects', '#']
  )

  new_fs = {}
  pred_answers = []
  i=0
  for resp in resps:
    all_vars = merge_dicts([context_vars, new_fs])
    new_fs.update(create_new_fs_from_code(prompt_f_gen, resp, all_vars))
    
    all_vars = merge_dicts([context_vars, new_fs])
    lvars = {}
    i=i+1
    print("executing response from {}th question".format(i))
    if i==5:
      print("response: ", resp)
    exec_safe(resp, all_vars, lvars)
    print("lvars: ", lvars)
    dict_keys = list(lvars.keys())
    if 'ret_val' in dict_keys:
      pred_answers.append(lvars['ret_val'])
    else:
      pred_answers.append("None")

  successes = []
  for resp, answer, pred_answer in zip(resps, answers, pred_answers):
    success = False
    try: 
      np.allclose(answer, pred_answer)
      success = True
    except Exception as e:
      print(e)
    successes.append(success)
  return successes, pred_answers, resps



#set the objects and their positions as a dictionary
obj_state = {
  'cyan bowl': np.array([0.5, 0.2]),
  'red bowl': np.array((0.2, 0.7)),
  'blue bowl': np.array((0.7, 0.5)),
  'red block': np.array((0.5, 0.25)),
  'cyan block': np.array((0.1, 0.8)),
  'blue block': np.array((0.3, 0.2)),
}

tabletop_coords = {
        'top_left':     ([]),
        'top_side':     ([]),
        'top_right':    ([]),
        'left_side':    ([]),
        'middle':       ([0.5 , 0.5]),
        'right_side':   (0.3 - 0.05,  -0.5,      ),
        'bottom_left':  (-0.3 + 0.05, -0.8 + 0.05),
        'bottom_side':  (0,           -0.8 + 0.05),
        'bottom_right': (0.3 - 0.05,  -0.8 + 0.05)
}


def get_obj_pos(name):
  return obj_state[name]

def get_corner_pos():
   
  return
   
def get_side_pos():
    
  return
   
def put_first_on_second():
      
  return




#-------------------simple single-step tasks-------------------------------------#

questions_1 =[
  '# put the red block on the blue bowl.',
  '# put the blue block on the red bowl.',
  '# put the cyan block on the cyan bowl.',
  '# put the red block on the red bowl.',

  '# put the blue block in the top left corner.',
  '# put the cyan block in the top right corner.',
  '# put the red block in the bottom left corner.',
  '# put the blue block in the bottom right corner.',
]

#-------------------complex single-step tasks-------------------------------------#

questions_2 = [
  '# put the blue block on the closest bowl to it.',
  '# put the cyan block on the closest bowl to it.',

  '# put the blue block on the furthest bowl to it.',
  '# put the red block on the furthest bowl to it.',

  '# put the red block in the closest corner to it.',
  '# put the cyan block in the closest corner to it.',

  '# move the blue block 5cm to its left.',
  '# move the red block 5cm to its right.',

  '# move the cyan block to the left side of the table.',
  '# move the red block to the right side of the table.',
]

#---------------------------multi-step tasks-------------------------------------#

questions_3 = [
  '# stack the blocks together.',
  '# move the blocks to the left side of the table.',
  '# move the blocks to the right side of the table.',
  '# move the blocks to the top left corner.',
  '# move the blocks to the bottom right corner.',
  '# group the objects with same color together.'
]

#--------------------------------------------------------------------------------#
prompt_f_gen = '''
  import numpy as np
  from shapely.geometry import *
  from shapely.affinity import *

  # define function: pt_np = move_pt_left(pt_np, dist).
  def move_pt_left(pt_np, dist):
      delta = np.array([-dist, 0])
      return translate_pt_np(pt_np, delta=delta)

  # define function: pt_np = move_pt_up(pt_np, dist).
  def move_pt_up(pt_np, dist):
      delta = np.array([0, dist])
      return translate_pt_np(pt_np, delta=delta)

  # example: interpolate a line at the halfway point.
  pt = np.array(line.interpolate(t, normalized=True).coords[0])
  '''.strip()

prompt_parse_positions = '''
  import numpy as np
  from shapely.geometry import *
  from shapely.affinity import *
  from utils import get_obj_pos, get_corners, get_sides

  # the top right corner.
  top_left_pos = np.array([0, 1])
  ret_val = top_left_pos
  # the bottom side.
  bottom_pos = np.array([0.5, 0])
  ret_val = bottom_pos
  # point 0.1 right of the red bowl.
  ret_val = get_obj_pos('red bowl') + [0.1, 0]
  # a line with 3 points from the blue block to the bottom right corner.
  start_pos = get_obj_pos('blue block')
  end_pos = np.array([1, 0])
  line = make_line(start=start_pos, end=end_pos)
  points_np = interpolate_pts_on_line(line=line, n=3)
  ret_val = points_np
  # a circle with 10 points around the center with radius 0.1.
  circle = make_circle(center=np.array([0.5, 0.5]), radius=0.1)
  pts_coords = interpolate_pts_along_exterior(exterior=circle.exterior, n=10)
  ret_val = pts_coords
  # loop that goes through all the sides.
  sides = get_sides()
  loop = np.r_[sides, [sides[0]]]
  ret_val = loop
  '''.strip()

context_vars = {
      'np': np,
      'get_corners': get_corners,
      'get_sides': get_sides ,
      'get_obj_pos': get_obj_pos,
      'put_first_on_second': put_first_on_second
}

for questions in[questions_1, questions_2, questions_3]:
  successes, pred_answers, resps = eval_questions_lmp(prompt_f_gen, prompt_parse_positions, questions, answers, context_vars)
  print("successes: ", successes)
  print('Success Rate: ', np.mean(successes))
  message = "successes: " + str(successes)
  message += "success rate: " + str(np.mean(successes))
  message +="--------------------------------------------------------------------"
  save_to_file(file_path, message)

  if append_failure:
    message = ""
    message += 'Failures\n\n'
    for s, q, r, pa, a in zip(successes, questions_short, resps, pred_answers, answers):
      if not s:
        msg = f"question: {q}\n \
        response: {r}\n \
        predicted answer: {pa}\n \
        actual answer: {a}\n \
        ------------------------------------------------------------------\n\n"
        message += msg
    save_to_file(file_path, message)






















def put_first_on_second(self, arg1, arg2):
  # put the object with obj_name on top of target
  # target can either be another object name, or it can be an x-y position in robot base frame
  pick_pos = self.get_obj_pos(arg1) if isinstance(arg1, str) else arg1
  place_pos = self.get_obj_pos(arg2) if isinstance(arg2, str) else arg2
  self.env.step(action={'pick': pick_pos, 'place': place_pos})


def get_corner_positions(self):
  normalized_corners = np.array([
      [0, 1],
      [1, 1],
      [0, 0],
      [1, 0]
  ])
  return np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))

def get_side_positions(self):
  normalized_sides = np.array([
      [0.5, 1],
      [1, 0.5],
      [0.5, 0],
      [0, 0.5]
  ])
  return np.array(([self.denormalize_xy(side) for side in normalized_sides]))

def get_corner_name(self, pos):
  corner_positions = self.get_corner_positions()
  corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
  return ['top left corner', 'top right corner', 'bottom left corner', 'botom right corner'][corner_idx]

def get_side_name(self, pos):
  side_positions = self.get_side_positions()
  side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
  return ['top side', 'right side', 'bottom side', 'left side'][side_idx]

#load prompts from file
import os
prompt_names = ['prompt_tabletop_ui','prompt_parse_position','prompt_parse_obj_name','prompt_parse_question','prompt_fgen','prompt_transform_shape_pts']
for name in prompt_names:
    file_name =  name + '_ori'
    save_path ='/home/hanl/master_thesis/code_as_policies/prompts'
    file1 = open(os.path.join(save_path, file_name), "r")
    exec(name + ' = file1.read()')
    file1.close()

prompt_tabletop_ui ="""
# Python 2D robot control script
import numpy as np
from env_utils import put_first_on_second, get_obj_pos, get_obj_names, say, get_corner_name, get_side_name, is_obj_visible, stack_objects_in_order
from plan_utils import parse_obj_name, parse_position, parse_question

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# place the yellow block on the yellow bowl.
say('Ok - putting the yellow block on the yellow bowl')
put_first_on_second('yellow block', 'yellow bowl')

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# move the green block to the top right corner.
say('Got it - putting the green block on the top right corner')
corner_pos = parse_position('top right corner')
put_first_on_second('green block', corner_pos)

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# stack the blue bowl on the yellow bowl on the green block.
order_bottom_to_top = ['green block', 'yellow block', 'blue bowl']
say(f'Sure - stacking from top to bottom: {", ".join(order_bottom_to_top)}')
stack_objects_in_order(object_names=order_bottom_to_top)

objects = ['cyan block', 'white block', 'cyan bowl', 'blue block', 'blue bowl', 'white bowl']
# move the cyan block into its corresponding bowl.
matches = {'cyan block': 'cyan bowl'}
say('Got it - placing the cyan block on the cyan bowl')
for first, second in matches.items():
  put_first_on_second(first, get_obj_pos(second))

objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# put the small banana colored thing in between the blue bowl and green block.
say('Sure thing - putting the yellow block between the blue bowl and the green block')
target_pos = parse_position('a point in the middle betweeen the blue bowl and the green block')
put_first_on_second('yellow block', target_pos)

objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# can you cut the bowls in half.
say('no, I can only move objects around')

objects = ['yellow block', 'green block', 'yellow bowl', 'gray block', 'gray bowl', 'green bowl']
# stack the blocks on the right side with the gray one on the bottom.
say('Ok. stacking the blocks on the right side with the gray block on the bottom')
right_side = parse_position('the right side')
put_first_on_second('gray block', right_side)
order_bottom_to_top = ['gray block', 'green block', 'yellow block']
stack_objects_in_order(object_names=order_bottom_to_top)

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# move the grass-colored bowl 10cm to the left.
say('Sure - moving the green bowl left by 10 centimeters')
left_pos = parse_position('a point 10cm left of the green bowl')
put_first_on_second('green bowl', left_pos)

""".strip()


#print word count in each prompt
prompt_names = ['prompt_tabletop_ui','prompt_parse_position','prompt_parse_obj_name','prompt_parse_question','prompt_fgen','prompt_transform_shape_pts']
prompt_word_count={}
for name in prompt_names:
    prompt_word_count[name] = len(eval(name).split(' '))
    print(name, prompt_word_count[name])

temperature = input("Please enter temperature (0.0-1.0): ")
if float(temperature) > 1.0 or float(temperature) < 0.0:
    raise ValueError("Invalid input. Please enter a value between 0.0 and 1.0. ")
else:
    temperature = float(temperature)

cfg_tabletop = {
  'lmps': {
    'tabletop_ui': {
      'prompt_text': prompt_tabletop_ui,
      'engine': 'text-davinci-003',
      'max_tokens': 512,
      'temperature': temperature,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': True,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_obj_name': {
      'prompt_text': prompt_parse_obj_name,
      'engine': 'text-davinci-003',
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_position': {
      'prompt_text': prompt_parse_position,
      'engine': 'text-davinci-003',
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_question': {
      'prompt_text': prompt_parse_question,
      'engine': 'text-davinci-003',
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'transform_shape_pts': {
      'prompt_text': prompt_transform_shape_pts,
      'engine': 'text-davinci-003',
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'new_shape_pts',
    },
    'fgen': {
      'prompt_text': prompt_fgen,
      'engine': 'text-davinci-003',
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# define function: ',
      'query_suffix': '.',
      'stop': ['# define', '# example'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
    }
  }
}

lmp_tabletop_coords = {
        'top_left':     (-0.3 + 0.05, -0.2 - 0.05),
        'top_side':     (0,           -0.2 - 0.05),
        'top_right':    (0.3 - 0.05,  -0.2 - 0.05),
        'left_side':    (-0.3 + 0.05, -0.5,      ),
        'middle':       (0,           -0.5,      ),
        'right_side':   (0.3 - 0.05,  -0.5,      ),
        'bottom_left':  (-0.3 + 0.05, -0.8 + 0.05),
        'bottom_side':  (0,           -0.8 + 0.05),
        'bottom_right': (0.3 - 0.05,  -0.8 + 0.05),
        'table_z':       0.0,
      }

def setup_LMP(env, cfg_tabletop):
  # LMP env wrapper
  cfg_tabletop = copy.deepcopy(cfg_tabletop)
  cfg_tabletop['env'] = dict()
  cfg_tabletop['env']['init_objs'] = list(env.obj_name_to_id.keys())
  cfg_tabletop['env']['coords'] = lmp_tabletop_coords
  LMP_env = LMP_wrapper(env, cfg_tabletop)
  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np
  }
  fixed_vars.update({
      name: eval(name)
      for name in shapely.geometry.__all__ + shapely.affinity.__all__
  })
  variable_vars = {
      k: getattr(LMP_env, k)
      for k in [
          'get_bbox', 'get_obj_pos', 'get_color', 'is_obj_visible', 'denormalize_xy',
          'put_first_on_second', 'get_obj_names',
          'get_corner_name', 'get_side_name',
      ]
  }
  variable_vars['say'] = lambda msg: print(f'robot says: {msg}')

  # creating the function-generating LMP
  lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars)

  # creating other low-level LMPs
  variable_vars.update({
      k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
      for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
  })

  # creating the LMP that deals w/ high-level language commands
  lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars
  )

  return lmp_tabletop_ui


