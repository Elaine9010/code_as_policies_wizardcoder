import os
import json
from code_as_policy import *
from vllm import LLM, SamplingParams
from moviepy.editor import ImageSequenceClip
from pprint import pprint

#TODO: clean code 
# load all prompts
with open("configs/prompts.json", 'r') as f:
    promptsDict = json.load(f)

# Creating variables dynamically
for key, value in promptsDict.items():
    globals()[key] = value

#print word count in each prompt
prompt_names = ['prompt_tabletop_ui_sim','prompt_parse_position','prompt_parse_obj_name','prompt_parse_question','prompt_fgen','prompt_transform_shape_pts']
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
    'prompt_text': prompt_tabletop_ui_sim,
    'engine': 'text-davinci-003',
    'max_tokens': 512,
    'temperature': temperature,
    'query_prefix': '# ',
    'query_suffix': '.',
    'stop': ['#', 'objects = ['],
    'maintain_session': True,
    'debug_mode': False,
    'include_context': True,
    'has_return': False,
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


input_size = input("Please enter a value (7B or 13B): ")
if input_size in ["7B", "13B"]:
    model_size=input_size
else:
    raise ValueError("Invalid input. Please enter either 7B or 13B. ")

base_model="WizardLM/WizardCoder-Python-{f}-V1.0".format(f=model_size)
llm = LLM(model=base_model, tensor_parallel_size=1)

#@title Initialize Env { vertical-output: true }
num_blocks = 3 #@param {type:"slider", min:0, max:4, step:1}
num_bowls = 3 #@param {type:"slider", min:0, max:4, step:1}
high_resolution = False #@param {type:"boolean"}
high_frame_rate = False #@param {type:"boolean"}

# setup env and LMP
env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
block_list = ['purple block', 'green block', 'gray block']
bowl_list = ['purple bowl', 'pink bowl', 'green bowl']
obj_list = block_list + bowl_list
_ = env.reset(obj_list)


#print("INFO: Intial postion of all objects: ", env.init_pos)
lmp_tabletop_ui = setup_LMP(env, llm, cfg_tabletop)

# display env
import matplotlib.pyplot as plt
plt.imshow(env.get_camera_image())
plt.show()
print('available objects:')
print(obj_list)


while True:
    # Get user input
    user_input = input("Please enter user input: (terminate the session by entering 'exit')")
    
    env.cache_video = []

    if user_input.strip() == 'exit':
        break
    
    print('Running policy and recording video...')
    try:
        ret_val = lmp_tabletop_ui(user_input, f'objects = {env.object_list}')
    except Exception as e:
        print("WARNING: LMP failed to execute. ", e)
        continue

    # get all object pos
    init_obj_pos = env.init_pos
    current_obj_pos = {objname : env.get_obj_pos(objname) for objname in obj_list}
    print("inital object pos: \n")
    pprint(init_obj_pos)

    print("current object pos: \n")
    pprint(current_obj_pos)


    # render video
    output_file = 'output_video.mp4'
    if env.cache_video:
        rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
        rendered_clip.write_videofile(output_file, codec='libx264')
        print(f'Video saved to {output_file}')
