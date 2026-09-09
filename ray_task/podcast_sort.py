import os
import json
import traceback
from typing import List, Dict

from utils.logger import Logger
logger = Logger.get_logger()

current_directory = os.path.dirname(os.path.abspath(__file__))
podcast_sort_config_file = os.path.join(current_directory, "podcast_sort.json")


def has_file_changed(file_path, last_mtime):
    try:
        current_mtime = os.path.getmtime(file_path)
        if current_mtime != last_mtime:
            return True, current_mtime
    except Exception as e:
        logger.error(f"func has_file_changed exception {traceback.format_exc()}")

    return False, 0

def move_prefix_to_front(lst: List, prefix: str):
    prefix_elements = [s for s in lst if s.startswith(prefix)]
    for element in prefix_elements:
        lst.remove(element)
    lst[0:0] = prefix_elements

last_sort_mtime = 0
def sort_podcast_todo_tasks(todo_tasks: List[Dict]):
    global last_sort_mtime
    if not os.path.exists(podcast_sort_config_file):
        return

    file_changed, changed_mtime = has_file_changed(podcast_sort_config_file, last_sort_mtime)
    if file_changed:
        last_sort_mtime = changed_mtime
        sort_config = {}
        with open(podcast_sort_config_file, 'r', encoding='utf-8') as f:
            sort_config = json.load(f)

        if "high_priority_podcast" in sort_config and sort_config["high_priority_podcast"]:
            high_priority_podcast: List[str] = sort_config["high_priority_podcast"]
            move_elements = [s for s in todo_tasks if s["podcast_name"] in high_priority_podcast]
            for element in move_elements:
                todo_tasks.remove(element)
            todo_tasks[0:0] = move_elements
   

if __name__ == "__main__":
    todo_tasks = [
        {
            "podcast_name": "李丁聊天室",
            "episode_name": "300567_32. 美国工程师兼职做房产经纪人(嘉宾：何仁).mp3",
            "audio_duration_second": 2195
        },
        {
            "podcast_name": "李丁聊天室",
            "episode_name": "300545_55. 中美职场沟通差异和技巧 ｜ 用三明治沟通法表达否定 ｜ 怎么更加高效的 1 on 1(嘉宾：陈然).mp3",
            "audio_duration_second": 2878
        },
        {
            "podcast_name": "播客相对论 ｜ 每周推荐值���收藏的播客单集",
            "episode_name": "5123690_Vol.15 让你的听觉更懂视觉.m4a",
            "audio_duration_second": 464
        },
        {
            "podcast_name": "test_sort",
            "episode_name": "5123685_Vol.20 给未来留一个记忆胶囊.m4a",
            "audio_duration_second": 531
        },
        {
            "podcast_name": "播客相对论 ｜ 每周推荐值得收藏的播客单集",
            "episode_name": "5123692_二月份不能错过的播客单集.m4a",
            "audio_duration_second": 544
        },
        {
            "podcast_name": "播客相对论 ｜ 每周推荐值得收藏的播客单集",
            "episode_name": "5123685_Vol.20 给未来留一个记忆胶囊.m4a",
            "audio_duration_second": 531
        },
    ]
    sort_podcast_todo_tasks(todo_tasks)
    print(todo_tasks)