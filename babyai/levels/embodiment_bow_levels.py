
import numpy as np
import gym
from gym_minigrid.minigrid import GeneralObj

from .verifier import *
from .levelgen import *


class BOWLevelBase(RoomGridLevel):
    def __init__(self, room_size=11, num_dists=8, seed=None, **kwargs):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def add_object(self, i, j, kind, color):
        """
        Add a new object to room (i, j)
        """
        obj = GeneralObj(kind, color)

        return self.place_in_room(i, j, obj)



class Level_BOWPickUpGoToLocalShapeSplits(BOWLevelBase):
    def __init__(self, room_size=11, num_dists=8, seed=None, training = True):
        self.shapes1 = ['basketball', 'book', 'bottle', 'candle', 'comb', 'cube', 'block', 'cushion', 'fork', 'football', 'glass', 'drill', 'gun', 'phone', 'knife', 'mug', 'napkin', 'pen', 'pencil', 'picture', 'pillar', 'plate', 'plant', 'tile', 'duck', 'scissors', 'soap', 'cup', 'sponge', 'spoon', 'lamp']
        self.shapes2 = ['boat', 'bus', 'car', 'carriage', 'helicopter', 'keyboard', 'plane', 'robot', 'rocket', 'train', 'bat', 'vase']
        self.all_shapes = self.shapes1 + self.shapes2
        
        if training:
            self.pickupshapes = self.shapes1
        else:
            self.pickupshapes = self.shapes2
        
        self.colors = ['grey', 'blue', 'green', 'black', 'orange', 'purple', 'pink', 'red', 'white', 'yellow']
        self.training = training

        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed)

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True, at_least_1shape = []):
        """
        Add random objects, with at least one object has one of the guaranteed shapes
        """
        COLOR_NAMES = self.colors

        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        guaranteed = False
        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            if not guaranteed and len(at_least_1shape) > 0:
                objtype = self._rand_elem(at_least_1shape)
                guaranteed = True
            else:                
                objtype = self._rand_elem(self.all_shapes)
            obj = (objtype, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i == None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j == None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists

    def add_shapes_select_target(self, target_from):
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=True, at_least_1shape = target_from)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        while obj.type not in target_from:
            obj = self._rand_elem(objs)

        return obj

    def gen_mission(self):
        self.place_agent()
        if self.training:
            task = self._rand_elem(['pickup', 'goto'])
        else:
            task = 'pickup'

        if task == 'pickup':
            target = self.add_shapes_select_target(target_from = self.pickupshapes)
            self.instrs = PickupInstr(ObjDesc(target.type), strict = True)
        else:
            target = self.add_shapes_select_target(target_from = self.all_shapes)
            self.instrs = GoToInstr(ObjDesc(target.type))


class Level_BOWPickUpGoToLocalShapeSplitsTest(Level_BOWPickUpGoToLocalShapeSplits):
    def __init__(self, room_size=11, num_dists=8, seed=None):
        super().__init__(room_size = room_size, num_dists = num_dists, seed = seed, training = False)


# Register the levels in this file
register_levels(__name__, globals(), prefix = 'Embodiment')