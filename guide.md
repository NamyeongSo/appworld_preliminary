사용 방법:

cd /home/thskadud/appworld
source .appworld/bin/activate
export APPWORLD_ROOT=/home/thskadud/appworld

appworld --help
appworld verify tests

Python에서:

from appworld import AppWorld, load_task_ids

task_id = load_task_ids("dev")[0]