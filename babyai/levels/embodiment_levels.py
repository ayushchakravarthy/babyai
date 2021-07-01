
import itertools

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
    generator = None
    def __init__(self, room_size=8, num_dists=3, seed=None):
        self.batch_size = 120 # need to be synced manually with batch_evaluate
        self.generator_id = 0
        super().__init__(room_size = room_size, num_dists = num_dists, seed = seed, training = False, max_steps = 16)

    def build_generator(self):
        csgen = [('ball', c, dc, dshape) for dshape in ['box', 'key'] for c in self.ball_colors for dc in self.all_colors if dc != c]
        csgen += [('box', c, dc, dshape) for dshape in ['ball', 'key'] for c in self.box_colors for dc in self.all_colors if dc != c]
        ColorSplitsTestBase.generator = list(itertools.product(csgen, range(3), range(2)))

    def seed(self, seed):
        if seed is None:
            self.generator_id = 0
        else:
            self.generator_id = seed - int(1e9)
        return super().seed(seed)

    def add_distractors(self):
        if ColorSplitsTestBase.generator is None:
            self.build_generator()

        if self.generator_id >= len(ColorSplitsTestBase.generator):
            self.generator_id %= len(ColorSplitsTestBase.generator)
            print('Generator exhausted')

        (tgt_shape, tgt_color, dcolor, dshape), tgt_loc, dselect = ColorSplitsTestBase.generator[self.generator_id]
        self.generator_id += self.batch_size        

        getobj = lambda shape, color: WorldObj.decode(*WorldObj(shape, color).encode())
        
        target = getobj(tgt_shape, tgt_color)
        d1 = getobj(tgt_shape, dcolor)
        d2 = getobj(dshape, tgt_color)

        objs = [None, None, None]
        distractors = [d1, d2]
        objs[tgt_loc] = target
        objs[0 if tgt_loc != 0 else 1] = distractors[dselect]
        objs[2 if tgt_loc != 2 else 1] = distractors[1-dselect]
        
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
        pos = np.array((4,6))
        self.grid.set(*pos, None)
        self.agent_pos = pos
        self.agent_dir = 3

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


class Level_PutNextLocalShapeSplits(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """

    def __init__(self, room_size=8, num_objs=8, seed=None, training = True):
        if training:
            self.o1types = {'box', 'ball'}
            self.o2types = {'key'}
        else:
            self.o1types = {'key'}
            self.o2types = {'box', 'ball'}

        self.all_shapes = self.o1types | self.o2types

        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True, guaranteed_shapes = []):
        """
        Add random objects, with at least one object has for each of the guaranteed shapes
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

        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            if len(guaranteed_shapes) > 0:
                objtype = guaranteed_shapes.pop()
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

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True, guaranteed_shapes = ['key', 'box', 'ball'])
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)
        while o1.type not in self.o1types:
            o1 = self._rand_elem(objs)
        while o2.type not in self.o2types or o1 == o2:
            o2 = self._rand_elem(objs)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PutNextLocalShapeSplitsTest(Level_PutNextLocalShapeSplits):
    def __init__(self, room_size=8, num_objs=8, seed=None):
        super().__init__(room_size=room_size, num_objs = num_objs, seed=seed, training = False)


# Register the levels in this file
register_levels(__name__, globals(), prefix = 'Embodiment')