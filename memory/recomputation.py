import re
import networkx as nx
from logger_utils import SingleLogger


class CheckpointsSelector:
    @classmethod
    def get_checkpoint_selector(cls, mode):
        if mode == "speed":
            return SpeedCheckpointsSelector()
        elif mode == "memory":
            return MemoryCheckpointsSelector()
        else:
            raise ValueError("%s is not found" % mode)

    @staticmethod
    def select_checkpoints(schedule):
        raise NotImplementedError


class SpeedCheckpointsSelector(CheckpointsSelector):
    @staticmethod
    def select_checkpoints(schedule):
        return list(filter(lambda n: len(re.findall("conv2d|conv|matmul", n.op))
                    > 0, schedule.operators))


class MemoryCheckpointsSelector(CheckpointsSelector):
    @staticmethod
    def select_checkpoints(schedule):
        # TODO(yuchen): https://arxiv.org/pdf/1604.06174.pdf
        raise NotImplementedError


def get_recomputation_edited_graph(dag, schedule, mode, verbose=True):
    selector = CheckpointsSelector.get_checkpoint_selector(mode)
    checkpoints = selector.select_checkpoints(schedule)
    if not checkpoints:
        SingleLogger().warn("No checkpoints found! Recomputation Aborted!")
        return False

    if verbose:
        names = [node.name for node in checkpoints]
        SingleLogger().info("select %d checkpoints: %s" %
                            (len(names), ', '.join(names)))

    _apply_recomputation(dag, schedule, checkpoints)

    return True


def _update_schedule(schedule, checkpoints):
    name_to_checkpoints = {node.name: node for node in checkpoints}
    for op in schedule.operators:
        if op.name in name_to_checkpoints:
            op.requires_grad = True
        else:
            op.requires_grad = False


def _update_dag(dag, checkpoints):
    pass


def _apply_recomputation(dag, schedule, checkpoints):
    _update_schedule(schedule, checkpoints)
    _update_dag(dag, checkpoints)
