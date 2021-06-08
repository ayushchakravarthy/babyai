
import numpy as np
import gym
from gym_minigrid.minigrid import COLORS, WorldObj

from .verifier import *
from .levelgen import *


class ColorSplitsBase(RoomGridLevel):
    def __init__(self, room_size=8, num_dists=8, seed=None, box_colors = ['red', 'green', 'blue'], ball_colors = ['purple', 'yellow', 'grey'], training = True, baseline = False, **kwargs):
        self.num_dists = num_dists

        # Non-intersecting color sets for boxes and balls, all colors for keys
        self.training = training
        self.baseline = baseline
        self.all_colors = set(COLORS.keys())
        if self.training:
            self.box_colors = box_colors
            self.ball_colors = ball_colors
            if not self.baseline:
                self.shapes = ['key', 'ball', 'box']
            else:
                self.shapes = ['key']
        else:
            self.box_colors = list(self.all_colors - set(box_colors))
            self.ball_colors = list(self.all_colors - set(ball_colors))
            self.shapes = ['ball', 'box']
        self.all_colors = list(self.all_colors)

        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
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


class ColorSplitsTestBase(ColorSplitsBase):
    def __init__(self, room_size=8, num_dists=3, seed=None):
        super().__init__(room_size = room_size, num_dists = num_dists, seed = seed, training = False, max_steps = 16)

    def add_distractors(self):
        tgt_shape = self._rand_elem(self.shapes)
        tgt_color = self.color_selector(tgt_shape)
        getobj = lambda shape, color: WorldObj.decode(*WorldObj(shape, color).encode())
        target = getobj(tgt_shape, tgt_color)
        
        dcolor = self._rand_elem(self.all_colors)
        while dcolor == tgt_color:
            dcolor = self._rand_elem(self.all_colors)
        d1 = getobj(tgt_shape, dcolor)

        dshape = self._rand_elem(self.shapes)
        while dshape == tgt_shape:
            dshape = self._rand_elem(self.shapes)
        d2 = getobj(dshape, tgt_color)

        objs = [target, d1, d2]
        self.np_random.shuffle(objs)
        locs = [(2,3), (4,3), (6,3)]

        room = self.get_room(0, 0)
        for i in range(3):
            pos = self.place_obj(objs[i], locs[i], (1,1), max_tries = 1)
            room.objs.append(objs[i])

        return target


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
    def __init__(self, room_size=8, num_dists=8, seed=None, training = True):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = training)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color), strict = True)


class Level_PickupLocalColorSplitsTest(Level_PickupLocalColorSplits):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, training = False)


class Level_PickupLocalColorSplitsTestStrict(ColorSplitsTestBase):
    def gen_mission(self):
        self.agent_pos = None
        pos = self.place_obj(None, (4,6), (1,1), max_tries=1)
        self.agent_pos = pos
        self.agent_dir = 0

        target = self.add_distractors()
        self.instrs = PickupInstr(ObjDesc(target.type, target.color), strict = True)


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

class ShapeColorGeneralizationBase(RoomGridLevel):
    def __init__(self, room_size=8, num_dists=8, seed=None, training = True, splits = None, baseinstr = GoToInstr, geninstr = lambda desc: PickupInstr(desc, strict = True), **kwargs):
        self.num_dists = num_dists
        self.training = training
        self.baseinstr = baseinstr
        self.geninstr = geninstr

        self.all_shapes = {'key', 'ball', 'box'}

        self.splits = splits
        if splits == 'color':
            self.common_shapes = set()
        elif splits == 'shape':
            self.common_shapes = {'key'}
        else:
            raise ValueError('Must be either color or shape generalization')

        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True, guaranteed_shapes = []):
        """
        Add random objects, with at least one object has one of the guaranteed shapes
        """
        COLOR_NAMES = list(COLORS.keys())

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
            if not guaranteed and len(guaranteed_shapes) > 0:
                objtype = self._rand_elem(guaranteed_shapes)
                guaranteed = True
            else:                
                objtype = self._rand_elem(list(self.all_shapes))
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

    def add_shapes_select_target(self, exclude_shapes = set()):
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False, guaranteed_shapes = list(self.all_shapes - exclude_shapes))
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        while obj.type in exclude_shapes:
            obj = self._rand_elem(objs)

        return obj

    def gen_mission(self):
        self.place_agent()

        if self.training:
            mode = self.np_random.choice(['base', 'gen'])

            if mode == 'gen':
                if self.splits == 'color':
                    obj = self.add_shapes_select_target()
                    self.instrs = self.geninstr(ObjDesc(type = None, color = obj.color))
                elif self.splits == 'shape':
                    obj = self.add_shapes_select_target(self.all_shapes - self.common_shapes)
                    self.instrs = self.geninstr(ObjDesc(obj.type))
            else:
                obj = self.add_shapes_select_target()
                self.instrs = self.baseinstr(ObjDesc(obj.type))
        else:
            obj = self.add_shapes_select_target(self.common_shapes)
            self.instrs = self.geninstr(ObjDesc(obj.type))


class Level_PickupGotoLocalColorSplits(ShapeColorGeneralizationBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, splits = 'color', training = True)


class Level_PickupGotoLocalColorSplitsTest(ShapeColorGeneralizationBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, splits = 'color', training = False)


class Level_PickupGotoLocalShapeSplits(ShapeColorGeneralizationBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, splits = 'shape', training = True)


class Level_PickupGotoLocalShapeSplitsTest(ShapeColorGeneralizationBase):
    def __init__(self, room_size=8, num_dists=8, seed=None):
        super().__init__(room_size=room_size, num_dists=num_dists, seed=seed, splits = 'shape', training = False)


# Register the levels in this file
register_levels(__name__, globals(), prefix = 'Embodiment')