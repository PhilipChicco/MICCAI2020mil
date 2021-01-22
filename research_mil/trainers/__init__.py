from .globaltaskmil_trainer import GlobalTaskMIL


trainers_dict = {
    'globaltaskmil'  : GlobalTaskMIL,
}

def get_trainer(name):
    names = list(trainers_dict.keys())
    if name not in names:
        raise ValueError('Invalid choice for trainers - choices: {}'.format(' | '.join(names)))
    return trainers_dict[name]