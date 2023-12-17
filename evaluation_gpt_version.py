import os
import json
import numpy as np
from code_as_policy import *
from vllm import LLM, SamplingParams
from moviepy.editor import ImageSequenceClip
from pprint import pprint
import openai


# parameters
model_size = "7B"
temperature = 0
dist_threshold = 0.03
prompt_version = "v1"
question_type = "2"

openai_api_key = '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'
openai.api_key = openai_api_key
model_name = 'text-davinci-002' # 'text-davinci-002'

"""
Object initial position with fixed seed (50)
'green block': [-0.00216038, -0.6087627 ,  0.02098936],
'red block': [-0.049075  , -0.3013662 ,  0.02098936],
'blue block': [ 0.16367035, -0.47601306,  0.02098936],
'green bowl': [-0.10702819, -0.45330745,  0.02      ],
'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]

corner positions
    "top left corner": (-0.25, -0.25, 0),
    "top side": (0, -0.25, 0),
    "top right corner": (0.25, -0.25, 0),
    "left side": (-0.25, -0.5, 0),
    "middle": (0, -0.5, 0),
    "right side": (0.25, -0.5, 0),
    "bottom left corner": (-0.25, -0.75, 0),
    "bottom side": (0, -0.75, 0),
    "bottom right corner": (0.25, -0.75, 0)
"""
# questions and desired object positions
questions1 = [
    "put the green block on the blue bowl",
    "place the red block on the green bowl",
    "put the blue block on the red bowl",

    "put the green block in the top left corner",
    "put the red block in the top right corner",
    "put the blue block in the bottom right corner",
]
questions2 =[
    "put the green block on the bowl closest to it",
    "put the red block on the bowl closest to it",
    "put the blue block on the bowl closest to it",

    "put the green block on the bowl farthest to it",
    "put the red block on the bowl farthest to it",
    "put the blue block on the bowl farthest to it",

    "put the green block on the corner closest to it",
    "put the red block on the corner closest to it",
    "put the blue block on the corner closest to it",

    "move the green block 5cm to its right",
    "move the red block 5cm to its left",
    "move the blue block 5cm to its left"
]
questions3 = [
    "move all the block to the left side of the table",
    "move all the block to the right side of the table",
    "put all the blocks in their corresponding bowls"
]
desired_objs_pos_list3 = [
    {
    'green block': [-0.25, -0.5, 0],
    'red block': [-0.25, -0.5, 0],
    'blue block': [ -0.25, -0.5, 0],
    'green bowl': [-0.10702819, -0.45330745,  0.02      ],
    'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
    'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
    'green block': [0.25, -0.5, 0],
    'red block': [0.25, -0.5, 0],
    'blue block': [0.25, -0.5, 0],
    'green bowl': [-0.10702819, -0.45330745,  0.02      ],
    'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
    'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
    'green block': [-0.10702819, -0.45330745,  0.02 ],
    'red block': [ 0.09410405, -0.32529953,  0.02 ],
    'blue block': [ -0.22282407, -0.6860539 ,  0.02  ],
    'green bowl': [-0.10702819, -0.45330745,  0.02      ],
    'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
    'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    }
]

desired_objs_pos_list2 = [
    {
        'green block': [-0.10702819, -0.45330745,  0.02      ],# closest green bowl
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [0.09410405, -0.32529953,  0.02     ],# closest red bowl
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [0.09410405, -0.32529953,  0.02      ],# closest red bowl
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [0.09410405, -0.32529953,  0.02   ],#furthest red bowl
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.22282407, -0.6860539 ,  0.02 ],# furthest blue bowl
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [-0.22282407, -0.6860539 ,  0.02      ],# furthest blue bowl
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.25, -0.75, 0],                      # closest corner bottom left
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.25, -0.25, 0],                         # closest corner top left
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [0.25, -0.25, 0],                         # closest corner top right
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [0.05-0.00216038, -0.6087627 ,  0.02098936], # 5cm to its right
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.099075  , -0.3013662 ,  0.02098936], # 5cm to its left
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.11367035, -0.47601306,  0.02098936], # 5cm to its left
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    }
]
desired_objs_pos_list1 = [
    {
        "green block": [-0.22282407, -0.6860539, 0.02],
        "red block": [-0.049075, -0.3013662, 0.02098936],
        "blue block": [0.16367035, -0.47601306, 0.02098936],
        "green bowl": [-0.10702819, -0.45330745, 0.02],
        "red bowl": [0.09410405, -0.32529953, 0.02],
        "blue bowl": [-0.22282407, -0.6860539, 0.02],
    },
    {
        "green block": [-0.00216038, -0.6087627, 0.02098936],
        "red block": [-0.10702819, -0.45330745, 0.02],
        "blue block": [0.16367035, -0.47601306, 0.02098936],
        "green bowl": [-0.10702819, -0.45330745, 0.02],
        "red bowl": [0.09410405, -0.32529953, 0.02],
        "blue bowl": [-0.22282407, -0.6860539, 0.02],
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.09410405, -0.32529953,  0.02      ],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.25, -0.25, 0],# top left corner
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },  
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [0.25, -0.25, 0],# top right corner
        'blue block': [ 0.16367035, -0.47601306,  0.02098936],
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    },
    {
        'green block': [-0.00216038, -0.6087627 ,  0.02098936],
        'red block': [-0.049075  , -0.3013662 ,  0.02098936],
        'blue block': [ 0.25, -0.75, 0],# bottom right corner
        'green bowl': [-0.10702819, -0.45330745,  0.02      ],
        'red bowl': [ 0.09410405, -0.32529953,  0.02      ],
        'blue bowl': [-0.22282407, -0.6860539 ,  0.02      ]
    }
]


# load questions and desired object positions
if question_type == "1":
    questions = questions1
    desired_objs_pos_list = desired_objs_pos_list1
elif question_type == "2":
    questions = questions2
    desired_objs_pos_list = desired_objs_pos_list2
elif question_type == "3":
    questions = questions3
    desired_objs_pos_list = desired_objs_pos_list3
else:
    raise ValueError("Invalid question type")

# load all prompts
with open(f"configs/prompts_{prompt_version}.json", "r") as f:
    promptsDict = json.load(f)

# Creating variables dynamically
for key, value in promptsDict.items():
    globals()[key] = value

# print word count in each prompt
prompt_names = [
    "prompt_tabletop_ui",
    "prompt_parse_position",
    "prompt_parse_obj_name",
    "prompt_parse_question",
    "prompt_fgen",
    "prompt_transform_shape_pts",
]
prompt_word_count = {}
for name in prompt_names:
    prompt_word_count[name] = len(eval(name).split(" "))
    print(name, prompt_word_count[name])

# cfg for tabletop
cfg_tabletop = {
    "lmps": {
        "tabletop_ui": {
            "prompt_text": prompt_tabletop_ui,
            "engine": "text-davinci-003",
            "max_tokens": 512,
            "temperature": temperature,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#"],
            "maintain_session": True,
            "debug_mode": False,
            "include_context": True,
            "has_return": False,
            "return_val_name": "ret_val",
        },
        "parse_obj_name": {
            "prompt_text": prompt_parse_obj_name,
            "engine": "text-davinci-003",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#","'''"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "parse_position": {
            "prompt_text": prompt_parse_position,
            "engine": "text-davinci-003",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#","def","import"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "parse_question": {
            "prompt_text": prompt_parse_question,
            "engine": "text-davinci-003",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "transform_shape_pts": {
            "prompt_text": prompt_transform_shape_pts,
            "engine": "text-davinci-003",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "new_shape_pts",
        },
        "fgen": {
            "prompt_text": prompt_fgen,
            "engine": "text-davinci-003",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# define function: ",
            "query_suffix": ".",
            "stop": ["# define", "# example"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
        },
    }
}



# setup env
high_resolution = False
high_frame_rate = False

# experiment objects list
block_list = ["green block", "red block", "blue block"]
bowl_list = ["green bowl", "red bowl", "blue bowl"]
obj_list = block_list + bowl_list

# success rate
successes = []
for idx, (question, desired_objs_pos) in enumerate(zip(questions, desired_objs_pos_list)):
    
    print(f"-----------------Question {idx+1}: {question}------------------")
    # setup env and LMP, TODO: deactive render
    env = PickPlaceEnv(
        render=True, high_res=high_resolution, high_frame_rate=high_frame_rate
    )
    _ = env.reset(obj_list)
    #openai version
    llm = None

    lmp_tabletop_ui = setup_LMP(env, llm, cfg_tabletop)

    try:
        ret_val = lmp_tabletop_ui(question, f"# objects = {env.object_list}")
    except Exception as e:
        print("WARNING: LMP failed to execute. ", e)
        successes.append(False)
        continue

    # calculate euclidean distance between current object pos and desired object pos
    max_dist = 0
    for obj in obj_list:
        current_obj_pos = env.get_obj_pos(obj)
        desired_obj_pos = desired_objs_pos[obj]
        dist = np.linalg.norm(np.array(current_obj_pos) - np.array(desired_obj_pos))
        #print(f"INFO: Euclidean distance between object({obj}) and desired object pos: {dist}")

        if dist > max_dist:
            max_dist = dist

    if max_dist < dist_threshold:
        successes.append(True)
    else:
        successes.append(False)

        # save failure video
        index =idx+1
        if not os.path.exists("failure_videos"):
            os.makedirs("failure_videos")
        output_file = f"failure_videos/failure_case_{index}.mp4"
        if env.cache_video:
            rendered_clip = ImageSequenceClip(
                env.cache_video, fps=35 if high_frame_rate else 25
            )
            rendered_clip.write_videofile(output_file, codec="libx264")
            print(f"Video saved to {output_file}")


print("INFO: Successes: ", successes,"  ",sum(successes),"/",len(successes))
# caluate success rate
success_rate = sum(successes) / len(successes)
print("Success rate: ", success_rate)


#save all results to log file
with open(f"cop_evaluation_result_gpt_log.txt","a") as f:
    f.write("--------------------------------------------------------------------------------------------------\n")
    f.write(f"model_size: {model_size}, temperature: {temperature}, dist_threshold: {dist_threshold}, prompt_version: {prompt_version}\n")
    f.write("questions:\n")
    f.write(f"{questions}\n")
    f.write(f"successes: {successes}\n")
    f.write(f"success_rate: {success_rate}\n")
    f.write("\n\n\n")