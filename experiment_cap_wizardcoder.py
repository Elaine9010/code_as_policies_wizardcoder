## this script evaluates the performance of wizardcoder with tasks from the paper "Code as Policies"
## task 1: select object by spatial descriptions / task 2: select positions by spatial descriptions
## case 1: direct language / case 2: chain of thought / case 3: LMP
## the success rate of each case is saved in the file "Evaluation_cop_wizardcoder_{model_size}_results.txt"

model_size = "7B"
temperature = 0
append_failure = False
select_cases = [ False, False, True]
default_query_kwargs = {
    'model_size': model_size,
    'max_tokens': 256,
    'temperature': temperature
}

from copy import copy
from tqdm.auto import trange
from time import sleep

import os
import ast
import astunparse

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from vllm import LLM, SamplingParams #wizardCoder
from datetime import datetime



#save evalutation results to a file
file_path = os.path.join(os.getcwd(), f"Evaluation_cop_wizardcoder_{model_size}_results_{temperature}.txt")
def save_to_file(file_path, result):
    file = open(file_path, "a")
    file.write(result)
    file.write("\n")
    file.close()
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
message = "--------------------------------" + current_time + "--------------------------------------"
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

def eval_safe(code_str, gvars=None, lvars=None):
  banned_phrases = ['import', '__']
  for phrase in banned_phrases:
    assert phrase not in code_str

  if gvars is None:
    gvars = {}
  if lvars is None:
    lvars = {}
  
  empty_fn = lambda *args, **kwargs: None
  custom_gvars = merge_dicts([
      gvars,
      {'exec': empty_fn, 'eval': empty_fn}
  ])
  return eval(code_str, custom_gvars, lvars)

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
    print("///////////////////DEBUG///////////////////////")
    print("code_str: ", code_str)
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

#--------------------------------------------------------------------------------------#
# # Task: Select Object by Spatial Descriptions

message = "task 1: select object by spatial descriptions"
print(message)
save_to_file(file_path, message)

query_setting_short = "objects = ['cyan bowl', 'red bowl', 'blue bowl', 'red block', 'cyan block', 'blue block']"

def get_obj_pos(name):
  return {
      'cyan bowl': np.array([0.5, 0.2]),
      'red bowl': np.array((0.2, 0.7)),
      'blue bowl': np.array((0.7, 0.5)),
      'red block': np.array((0.5, 0.25)),
      'cyan block': np.array((0.1, 0.8)),
      'blue block': np.array((0.3, 0.2)),
  }[name]

query_setting = '''
Setting: Objects and their positions are:
  - cyan bowl, (0.5, 0.2)
  - red bowl, (0.2, 0.7)
  - blue bowl, (0.7, 0.5)
  - red block, (0.5, 0.25)
  - cyan block, (0.1, 0.8)
  - blue block, (0.3, 0.2)
'''.strip()

questions_short = [
    'the top-most bowl',
    'the bottom-most bowl',
    'the left-most bowl',
    'the right-most bowl',
    'the top-most block',
    'the bottom-most block',
    'the left-most block',
    'the right-most block',
    'the second bowl from the right',
    'the second bowl from the left',
    'the second bowl from the top',
    'the second bowl from the bottom',
    'the second block from the right',
    'the second block from the left',
    'the second block from the top',
    'the second block from the bottom',
    'the block closest to the blue bowl',
    'the block closest from the red bowl',
    'the block closest to the cyan bowl',
    'the block farthest to the blue bowl',
    'the block farthest from the red bowl',
    'the block farthest to the cyan bowl',
    'the bowl closest to the blue block',
    'the bowl closest from the red block',
    'the bowl closest to the cyan block',
    'the bowl farthest to the blue block',
    'the bowl farthest from the red block',
    'the bowl farthest to the cyan block',
]

questions = [f'What is {q}?' for q in questions_short]

answers = [
    'red bowl', 'cyan bowl', 'red bowl', 'blue bowl',
    'cyan block', 'blue block', 'cyan block', 'red block',
    'cyan bowl', 'cyan bowl', 'blue bowl', 'blue bowl',
    'blue block', 'blue block', 'red block', 'red block',
    'red block', 'cyan block', 'red block',
    'cyan block', 'red block', 'cyan block', 
    'cyan bowl', 'cyan bowl', 'red bowl', 
    'red bowl', 'red bowl', 'cyan bowl'
]

import numpy as np

def eval_questions_lmp(prompt, query_setting_short, questions_short, answers, context_vars):
  queries = [
      f'{query_setting_short}\n# {question}.'
      for question in questions_short
  ]
  resps = lmp_batch(
      prompt,
      queries,
      strip=True,
      stop_tokens=['objects = [', '#']
  )

  pred_answers = []
  for resp in resps:
    lvars = {}
    print("///////////////////DEBUG///////////////////////")
    print("response: ", resp)
    print("context_vars: ", context_vars.keys())
    print("///////////////////DEBUG///////////////////////")
    exec_safe(resp, context_vars, lvars)
    pred_answers.append(lvars['ret_val'])

  successes = []
  for resp, answer, pred_answer in zip(resps, answers, pred_answers):
    successes.append(pred_answer == answer)
  return successes, pred_answers, resps

def eval_questions_language(prompt, query_setting, questions, answers):
  queries = [
      f'{query_setting}\nQuestion: {question}'
      for question in questions
  ]
  resps = lmp_batch(
      prompt,
      queries,
      strip=True,
      stop_tokens=['Setting', 'Question','Note'],
  )
  successes = []
  pred_answers = []
  for resp, answer in zip(resps, answers):
    pred_answer = resp[resp.find('Answer: ') + len('Answer: '):]
    successes.append(pred_answer == answer)
    pred_answers.append(pred_answer)
  return successes, pred_answers, resps



# ## Direct Language
if select_cases[0]:
  message = "case 1: direct language"
  print(message)
  save_to_file(file_path, message)

  prompt_parse_obj_name_nl = '''
  Setting: Objects and their positions are: 
    - blue block, (0, 0)
    - cyan block, (0.1, 0)
    - purple bowl, (0, 0.2)
    - gray bowl, (0.3, 0.4)
    - brown bowl, (0.5, 0.2)
    - purple block, (0.8, 0)
  Question: What is the block closest to the purple bowl?
  Answer: blue block
  Setting: Objects and their positions are:
    - brown bowl, (1, 0.5)
    - green block, (0.6, 0.8)
    - brown block, (0.32, 0.48)
    - green bowl, (0.1, 0.2)
    - blue bowl, (0.5, 0.5)
    - blue block, (0.4, 0.7)
  Question: What is the left most block?
  Answer: brown block
  Setting: Objects and their positions are:
    - brown bowl, (0.2, 0.3)
    - green block, (0.2, 0.35)
    - brown block, (0.1, 0.9)
    - green bowl, (0.4, 0.7)
    - blue bowl, (0.9, 0.2)
    - blue block, (0.7, 0.3)
  Question: What is the bowl near the top?
  Answer: green bowl
  '''

  successes, pred_answers, resps = eval_questions_language(prompt_parse_obj_name_nl, query_setting, questions, answers)
  print("successes: ", successes)
  print('Success Rate: ', np.mean(successes))

  o1 = np.mean(successes)
  message = "success rate: " + str(np.mean(successes))
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



# ## Chain of Thought
if select_cases[1]:
  message = "case 2: chain of thought"
  print(message)
  save_to_file(file_path, message)

  prompt_parse_obj_name_cot = '''
  Setting: Objects and their positions are: 
    - blue block, (0, 0)
    - cyan block, (0.1, 0)
    - purple bowl, (0, 0.2)
    - gray bowl, (0.3, 0.4)
    - brown bowl, (0.5, 0.2)
    - purple block, (0.8, 0)
  Question: What is the block closest to the purple bowl?
  Thought: The blocks and their positions are:
    - blue block, (0, 0)
    - cyan block, (0.1, 0)
    - purple block, (0.8, 0)
  Thought: Their distances to the purple bowl are:
    - blue block, 0.2
    - cyan block, 0.2236
    - purple block, 0.8246
  Thought: Therefore, the block closest to the purple bowl is the blue block.
  Answer: blue block
  Setting: Objects and their positions are:
    - brown bowl, (1, 0.5)
    - green block, (0.6, 0.8)
    - brown block, (0.32, 0.48)
    - green bowl, (0.1, 0.2)
    - blue bowl, (0.5, 0.5)
    - blue block, (0.4, 0.7)
  Question: What is the left most block?
  Thought: The blocks and their x coordinates are:
    - green block, 0.6
    - brown block, 0.32
    - blue block, 0.4
  Thought: The block with the minimum x coordinate is the brown block.
  Answer: brown block
  Setting: Objects and their positions are:
    - brown bowl, (0.2, 0.3)
    - green block, (0.2, 0.35)
    - brown block, (0.1, 0.9)
    - green bowl, (0.4, 0.7)
    - blue bowl, (0.9, 0.2)
    - blue block, (0.7, 0.3)
  Question: What is the bowl near the top?
  Thought: The bowls and their y coordinates are:
    - brown bowl, 0.3
    - green bowl, 0.7
    - blue bowl, 0.2
  Thought: The bowl with the highest y coordinate is the green bowl.
  Answer: green bowl
  '''

  successes, pred_answers, resps = eval_questions_language(prompt_parse_obj_name_cot, query_setting, questions, answers)
  print("successes: ", successes)
  print('Success Rate: ', np.mean(successes))

  o2 = np.mean(successes)
  message = "success rate: " + str(np.mean(successes))
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



# ## LMP
if select_cases[2]:
  message = "case 3: LMP"
  print(message)
  save_to_file(file_path, message)


  prompt_parse_obj_name_lmp = '''
  import numpy as np
  from env_utils import get_obj_pos

  objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block', 'purple block']
  # the block closest to the purple bowl.
  block_names = ['blue block', 'cyan block', 'purple block']
  block_positions = np.array([get_obj_pos(block_name) for block_name in block_names])
  closest_block_idx = np.argmin(np.linalg.norm(block_positions - get_obj_pos('purple bowl'), axis=1))
  closest_block_name = block_names[closest_block_idx]
  ret_val = closest_block_name
  objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
  # the left most block.
  block_names = ['green block', 'brown block', 'blue block']
  block_positions = np.array([get_obj_pos(block_name) for block_name in block_names])
  left_block_idx = np.argmin(block_positions[:, 0])
  left_block_name = block_names[left_block_idx]
  ret_val = left_block_name
  objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
  # the bowl on near the top.
  bowl_names = ['brown bowl', 'green bowl', 'blue bowl']
  bowl_positions = np.array([get_obj_pos(bowl_name) for bowl_name in bowl_names])
  top_bowl_idx = np.argmax(bowl_positions[:, 1])
  top_bowl_name = bowl_names[top_bowl_idx]
  ret_val = top_bowl_name
  '''.strip()

  context_vars = {
      'np': np,
      'get_obj_pos': get_obj_pos
  }
  successes, pred_answers, resps = eval_questions_lmp(prompt_parse_obj_name_lmp, query_setting_short, questions_short, answers, context_vars)
  print("successes: ", successes)
  print('Success Rate: ', np.mean(successes))
  message = "successes: " + str(successes)
  save_to_file(file_path, message)
  o3 = np.mean(successes)
  message = "success rate: " + str(np.mean(successes))
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



# # Task: Select Positions by Spatial Descriptions

message = "task 2: select positions by spatial descriptions"
print(message)
save_to_file(file_path, message)


from shapely import Point, box

def get_corners():
  return np.array([[0,0], [1, 0], [1,1], [0,1]])

def get_sides():
  return np.array([[0.5, 0], [0, 0.5], [1, 0.5], [0.5, 1]])

def get_obj_pos(name):
  return {
      'cyan bowl': np.array([0.6, 0.2]),
      'red bowl': np.array((0.2, 0.7)),
      'blue bowl': np.array((0.7, 0.6)),
      'red block': np.array((0.4, 0.25)),
      'cyan block': np.array((0.1, 0.8)),
      'blue block': np.array((0.3, 0.2)),
  }[name]

questions_short = [
    'a point 0.1 left of the cyan bowl',
    'a point 0.2 right of the red block',
    'a point 0.4 top of the blue bowl',
    'a point 0.1 bottom of the blue block',

    'the corner closest to the red block',
    'the corner closest to the blue bowl',
    'the corner farthest from the cyan block',
    'the corner farthest from the red bowl',

    'the side closest to the red block',
    'the side closest to the blue bowl',
    'the side farthest from the cyan block',
    'the side farthest from the red bowl',

    'a circle with 10 points centered around the red block with radius 0.2',
    'a circle with 12 points centered around the blue bowl with radius 0.3',

    'a line with 4 points from the top left corner to the bottom right corner',
    'a line with 5 points from the blue bowl to its closest side',
    'a line with 3 points from the blue block to its closest corner',
    'a line with 3 points from the middle to the red bowl',
    
    'a point in the middle of the red and blue blocks',
    'a point in the middle of the blue bowl and the corner closest to it',
    'a point in the middle between the top left corner and the bottom side',

    'a point in the middle of the blue, red, and cyan blocks',
    'a point in the middle of the blue, red, and cyan bowls',    
]

questions = [
    f'Find: {question_short}.' for question_short in questions_short
]

circ_red_block = Point(get_obj_pos('red block')).buffer(0.2)
circ_blue_bowl = Point(get_obj_pos('blue bowl')).buffer(0.3)
square_cyan_bowl = box(
    get_obj_pos('cyan bowl')[0] - 0.05, get_obj_pos('cyan bowl')[1] - 0.05,
    get_obj_pos('cyan bowl')[0] + 0.05, get_obj_pos('cyan bowl')[1] + 0.05,
  )

answers = [
    get_obj_pos('cyan bowl') + [-0.1, 0],
    get_obj_pos('red block') + [0.2, 0],
    get_obj_pos('blue bowl') + [0, 0.4],
    get_obj_pos('blue block') + [0, -0.1],

    np.array([0, 0]),
    np.array([1, 1]),
    np.array([1, 0]),
    np.array([1, 0]),

    np.array([0.5, 0]),
    np.array([1, 0.5]),
    np.array([1, 0.5]),
    np.array([1, 0.5]),

    np.array([circ_red_block.exterior.interpolate(t / 9, normalized=True).coords[0] for t in range(10)]),
    np.array([circ_red_block.exterior.interpolate(t / 11, normalized=True).coords[0] for t in range(12)]),

    np.linspace([0, 1], [1, 0], 4),
    np.linspace(get_obj_pos('blue bowl'), [1, 0.5], 5),
    np.linspace(get_obj_pos('blue block'), [0, 0], 3),
    np.linspace([0.5, 0.5], get_obj_pos('red bowl'), 3),

    (get_obj_pos('red block') + get_obj_pos('blue block')) / 2,
    (get_obj_pos('blue bowl') + np.array([1, 1])) / 2,
    (np.array([0, 1]) + np.array([0.5, 0])) / 2,

    (get_obj_pos('red block') + get_obj_pos('blue block') + get_obj_pos('cyan block')) / 3,
    (get_obj_pos('red bowl') + get_obj_pos('blue bowl') + get_obj_pos('cyan bowl')) / 3
]

def eval_questions_language(prompt, questions, answers):
  resps = lmp_batch(
      prompt,
      questions,
      strip=True,
      stop_tokens=['Setting', 'Find', 'Note'],
  )
  successes = []
  pred_answers = []
  for resp, answer in zip(resps, answers):
    pred_answer_str = resp[resp.find('Answer: ') + len('Answer: '):]
    try:
      pred_answer = np.array(eval_safe(pred_answer_str))
      success = np.allclose(pred_answer, answer, atol=1e-2)
    except Exception as e:
      print(e)
      success = False

    successes.append(success)
    pred_answers.append(pred_answer_str)
  return successes, pred_answers, resps

def eval_questions_lmp(prompt_f_gen, prompt, questions_short, answers, context_vars):
  queries = [
      f'\n# {question}.'
      for question in questions_short
  ]
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
    #i=i+1
    #print("executing response from {} th question".format(i))
    #if i==5:
    #  print("response: ", resp)

    print("///////////////////DEBUG///////////////////////")
    print("response: ", resp)
    print("context_vars: ", all_vars.keys())
    print("///////////////////DEBUG///////////////////////")
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



# # Direct Language
if select_cases[0]:
  message = "case 1: direct language"
  print(message)
  save_to_file(file_path, message)

  prompt_parse_pos_nl = '''
  Setting: Objects and their positions are: 
    - cyan bowl, (0.6, 0.2)
    - red bowl, (0.2, 0.7)
    - blue bowl, (0.7, 0.6)
    - red block, (0.4, 0.25)
    - cyan block, (0.1, 0.8)
    - blue block, (0.3, 0.2)
  Setting: The four corner names and their positions are:
    - top left corner, (0, 1)
    - top right corner, (1, 1)
    - bottom left corner, (0, 0)
    - bottom right corner, (1, 0)
  Setting: The four side names and their positions are:
    - top side, (0.5, 1)
    - right side, (1, 0.5)
    - bottom side, (0.5, 0)
    - left side, (0, 0.5)
  Find: the top right corner.
  Answer: (1, 1)
  Find: the bottom side.
  Answer: (0.5, 0)
  Find: a point 0.1 right of the red bowl.
  Answer: (0.3, 0.7)
  Find: a line with 3 points from the blue block to the bottom right corner.
  Answer: ((0.3, 0.2), (0.65, 0.1), (1, 0))
  Find: a circle with 10 points around the center with radius 0.1.
  Answer: ((0.6, 0.5), (0.577, 0.436), (0.517, 0.402), (0.450, 0.413), (0.406, 0.466), (0.406, 0.534), (0.450, 0.587), (0.517, 0.598), (0.577, 0.564), (0.6, 0.5))
  Find: a loop that goes through all the sides.
  Answer: ((0.5, 0), (0, 0.5), (1, 0.5), (0.5, 1), (0.5, 0))
  '''


  successes, pred_answers, resps = eval_questions_language(prompt_parse_pos_nl, questions, answers)
  print("successes: ", successes)
  print('Success Rate: ', np.mean(successes))

  p1 = np.mean(successes)
  message = "success rate: " + str(np.mean(successes))
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


# # Chain of Thought
if select_cases[1]:
  message = "case 2: chain of thought"
  print(message)
  save_to_file(file_path, message)

  prompt_parse_pos_cot = '''
  Setting: Objects and their positions are: 
    - cyan bowl, (0.6, 0.2)
    - red bowl, (0.2, 0.7)
    - blue bowl, (0.7, 0.6)
    - red block, (0.4, 0.25)
    - cyan block, (0.1, 0.8)
    - blue block, (0.3, 0.2)
  Setting: The four corner names and their positions are:
    - top left corner, (0, 1)
    - top right corner, (1, 1)
    - bottom left corner, (0, 0)
    - bottom right corner, (1, 0)
  Setting: The four side names and their positions are:
    - top side, (0.5, 1)
    - right side, (1, 0.5)
    - bottom side, (0.5, 0)
    - left side, (0, 0.5)
  Find: the top right corner.
  Answer: (1, 1)
  Find: the bottom side.
  Answer: (0.5, 0)
  Find: a point 0.1 right of the red bowl.
  Thought: The red bowl position is (0.2, 0.7).
  Thought: (0.2, 0.7) + (0.1, 0) = (0.3, 0.7)
  Answer: (0.3, 0.7)
  Find: a line with 3 points from the blue block to the bottom right corner.
  Thought: The start position is (0.3, 0.2).
  Thought: The end position is (1, 0).
  Thought: Interpolating one point in the middle is ((0.3, 0.2) + (1, 0)) / 2 = (0.65, 0.1)
  Answer: ((0.3, 0.2), (0.65, 0.1), (1, 0))
  Find: a circle with 10 points around the center with radius 0.1.
  Thought: The center position is (0.5, 0.5).
  Thought: The first and last point on the circle is (0.5, 0.5) + (0.1, 0) = (0.6, 0.5).
  Thought: Other points are obtained by applying increments of 360/10 = 36 degrees.
  Answer: ((0.6, 0.5), (0.577, 0.436), (0.517, 0.402), (0.450, 0.413), (0.406, 0.466), (0.406, 0.534), (0.450, 0.587), (0.517, 0.598), (0.577, 0.564), (0.6, 0.5))
  Find: a loop that goes through all the sides.
  Thought: The sides are ((0.5, 1), (1, 0.5), (0.5, 0), (0, 0.5)).
  Thought: A loop is a sequence that needs the first position in the last position.
  Answer: ((0.5, 1), (1, 0.5), (0.5, 0), (0, 0.5), (0.5, 1))
  '''

  successes, pred_answers, resps = eval_questions_language(prompt_parse_pos_cot, questions, answers)
  print("successes: ", successes)
  print('Success Rate: ', np.mean(successes))

  p2 = np.mean(successes)
  message = "success rate: " + str(np.mean(successes))
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


# # LMP
if select_cases[2]:
  message = "case 3: LMP"
  print(message)
  save_to_file(file_path, message)

  import shapely
  from shapely.geometry import *
  from shapely.affinity import *

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
      'get_sides': get_sides,
      'get_obj_pos': get_obj_pos
  }
  context_vars.update({
      name: getattr(shapely.geometry, name)
      for name in shapely.geometry.__all__
  })
  context_vars.update({
      name: getattr(shapely.affinity, name)
      for name in shapely.affinity.__all__
  })



  successes, pred_answers, resps = eval_questions_lmp(prompt_f_gen, prompt_parse_positions, questions_short, answers, context_vars)
  print("successes: ", successes)
  print('Success Rate: ', np.mean(successes))
  message = "successes: " + str(successes)
  save_to_file(file_path, message)
  p3 = np.mean(successes)
  message = "success rate: " + str(np.mean(successes))
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

#calculate total success rate and save to file
#for o,p,type_name in zip([o1,o2,o3],[p1,p2,p3],['vanilla' , 'chain of thought', 'lmp']):
#  t = (28*o + 23*p)/51
#  message = "Total success rate of" + type_name + " : "  + str(t)
# save_to_file(file_path, message)

t = (28*o3 + 23*p3)/51
message = "Total success rate of lmp : "  + str(t)
save_to_file(file_path, message)