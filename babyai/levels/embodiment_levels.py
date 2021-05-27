
import numpy as np
import gym
from .verifier import *
from .levelgen import *


class ColorSplitsBase(RoomGridLevel):
    def __init__(self, room_size=8, num_dists=8, seed=None, training = True, baseline = False):
        self.num_dists = num_dists

        # Non-intersecting color sets for boxes and balls, all colors for keys
        self.training = training
        self.baseline = baseline
        if self.training:
            self.box_colors = ['red', 'green', 'blue']
            self.ball_colors = ['purple', 'yellow', 'grey']
            if not self.baseline:
                self.shapes = ['key', 'ball', 'box']
            else:
                self.shapes = ['key']
        else:
            self.box_colors = ['purple', 'yellow', 'grey']
            self.ball_colors = ['red', 'green', 'blue']
            self.shapes = ['ball', 'box']
        self.all_colors = self.box_colors + self.ball_colors

        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def color_selector(self, obj_type):
        if obj_type == 'key':
            return self._rand_elem(self.all_colors)
        elif obj_type == 'box':
            return self._rand_elem(self.box_colors)
        elif obj_type == 'ball':
            return self._rand_elem(self.ball_colors)
        else:
            return

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True):
        """
        Add random objects that can potentially distract/confuse the agent.
        """

        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            obj_type = self._rand_elem(self.shapes)
            color = self.color_selector(obj_type)
            obj = (obj_type, color)

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

class Level_GotoLocalColorSplits(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = True)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GotoLocalColorSplitsTest(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = False)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GotoLocalColorSplitsBaseline(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(rroom_size=room_size, num_dists=num_dists, seed=seed, baseline = True)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        if obj.type == 'key':
            self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
        else:
            self.instrs = GoToInstr(ObjDesc(obj.type))

class Level_PickupLocalColorSplits(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = True)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_PickupLocalColorSplitsTest(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = False)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_PickupLocalColorSplitsBaseline(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, baseline = True)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        if obj.type == 'key':
            self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))
        else:
            self.instrs = PickupInstr(ObjDesc(obj.type))

class Level_PickupGotoLocalColorSplits(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=8, seed=None, training = True):
        self.rng = np.random.default_rng()
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = training)

    def gen_mission(self):
        mode = self.rng.choice(['Goto', 'Pickup'])
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        if mode == 'Goto':
            if obj.type == 'key':
                self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
            else:
                self.instrs = GoToInstr(ObjDesc(obj.type))
        else:
            self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_PickupGotoLocalColorSplitsTest(Level_PickupGotoLocalColorSplits):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = False)


# Register the levels in this file
register_levels(__name__, globals())