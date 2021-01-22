
from .globaltaskmil_test import GlobalTaskMILTest

testers_dict = {
    'globaltaskmil' : GlobalTaskMILTest,
}

def get_tester(name):
    names = list(testers_dict.keys())
    if name not in names:
        raise ValueError('Invalid choice for testers - choices: {}'.format(' | '.join(names)))
    return testers_dict[name]