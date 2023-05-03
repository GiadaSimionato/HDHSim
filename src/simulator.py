import pandas as pd
import numba as nb
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm
import os

from src.logger import GlobalLogger

class GlobalConfig:
    _cfg = None
    
    @classmethod
    def set_cfg(cls, cfg):
        assert cls._cfg == None
        cls._cfg = cfg
    
    @classmethod
    def cfg(cls):
        return cls._cfg

class RenderFunction:
    def __init__(self, *, fn):
        self.fn = fn
    
    def __call__(self, window, objects, state):
        self.fn(window=window, objects=objects, state=state)

class EventsFunction:
    def __init__(self, *, fn):
        self.fn = fn
    
    def __call__(self, objects, events, state):
        ret = self.fn(events=events, objects=objects, state=state)
        if ret is None: return dict()
        else: return ret

class GlobalFunction:
    def __init__(self, *, fn):
        self.fn = fn
    
    def __call__(self, objects, global_memory):
        return self.fn(objects=objects, global_memory=global_memory)

class TickFunction:
    def __init__(self, *, fn, backend="python", inputs=None, outputs=None, types=None):
        backends = ["python", "numba", "dask"]
        assert backend in backends, f"Available backends are {' '.join(backends)}"
        assert not inputs is None and not outputs is None, "You must specify inputs and outputs"
        self.fn = fn
        self.backend = backend
        self.inputs = inputs
        self.outputs = outputs

        self.meta = {k: "float" for k in self.inputs+self.outputs}
        if types is not None: self.meta.update(types)

        if backend == "numba":
            outputs_number = len(self.outputs)
            compiled_fn = nb.njit(fastmath=True)(fn)
            
            @nb.njit(fastmath=True, parallel=True)
            def numba_fn(objects):
                ret = np.zeros((len(objects), outputs_number))
                for i in nb.prange(len(objects)):
                    ret[i] = compiled_fn(objects[i])
                return ret
            
            self.fn = numba_fn


    def __call__(self, *args):
        if self.backend == "numba":
            objects, mask = args
            target_objects = objects[mask]
            fn_input = target_objects[self.inputs].to_numpy()
            fn_output = self.fn(fn_input)
            objects.loc[mask, self.outputs] = fn_output
            return objects
        elif self.backend == "python":
            objects, mask = args
            io = list(set(self.inputs+self.outputs))
            target_objects = objects[mask][io]
            results = target_objects.apply(self.fn, axis=1)
            objects.loc[mask, self.outputs] = results[self.outputs]
            return objects
        elif self.backend == "dask":
            objects, mask = args
            io = list(set(self.inputs+self.outputs))
            target_objects = objects[mask][io]
            target_dobjects = dd.from_pandas(target_objects, npartitions=os.cpu_count())
            meta = {c: self.meta[c] for c in target_dobjects.columns}
            results = target_dobjects.apply(self.fn, axis=1, meta=meta).compute()
            objects.loc[mask, self.outputs] = results[self.outputs]
            return objects


class Hook:
    def start(self, objects):
        raise NotImplementedError()
    
    def tick(self, objects):
        raise NotImplementedError()
    
    def end(self, objects):
        raise NotImplementedError()
    
    def msg(self, objects):
        raise NotImplementedError()
    

class Simulator:
    def __init__(self, cfg=None):
        GlobalConfig.set_cfg(cfg)
        self.objects = None
        self.tick_fns = []
        self.global_tick_fns = []
        self.render_fns = []
        self.events_fns = []
        self.hooks = []
        self.global_memory = dict()
        self.render_state = dict(_tick=0)
        self.running = True
        self.pause = False
    
    def add_objects(self, *, type, objects):
        assert isinstance(objects, pd.DataFrame)
        objects["type"] = type
        if self.objects is None:
            self.objects = objects.copy()
        else:
            self.objects = pd.concat((self.objects, objects))

        self.objects = self.objects.reset_index(drop=True)

    def add_global_tick_fn(self, *, function):
        assert isinstance(function, GlobalFunction), "Global functions must be GlobalFunction"
        self.global_tick_fns.append(function)
    
    def add_render_fn(self, *, function):
        assert isinstance(function, RenderFunction), "Render functions must be RenderFunction"
        self.render_fns.append(function)

    def add_events_fn(self, *, function):
        assert isinstance(function, EventsFunction), "Events functions must be EventsFunction"
        self.events_fns.append(function)
    
    def add_tick_fn(self, *, target, function):
        self.tick_fns.append(dict(target=target, function=function))

    def add_hook(self, hook):
        assert isinstance(hook, Hook)
        self.hooks.append(hook)

    def events(self):
        import pygame
        events = pygame.event.get()
        state_dict = dict()
        for events_fn in self.events_fns:
            ret = events_fn(objects=self.objects, events=events, state=self.render_state)
            state_dict.update(ret)

        if "quit" in state_dict and state_dict["quit"]:
            self.running = False
        
        for event in events:
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key == "space":
                    self.pause = not self.pause

    def render(self, *, window, clock, fps, background):
        import pygame
        window.fill(background)
        for render_fn in self.render_fns:
            render_fn(window=window, objects=self.objects, state=self.render_state)
        pygame.display.flip()
        if clock is not None: clock.tick(fps)

    def run_tick_fns(self):
        for fn in self.tick_fns:
            if fn["target"] != "*": mask = self.objects["type"] == fn["target"]
            else: mask = np.ones((len(self.objects)))

            self.objects = fn["function"](self.objects, mask)

    def run_global_tick_fns(self):
        for fn in self.global_tick_fns:
            self.objects = fn(objects=self.objects, global_memory=self.global_memory)

    def run_hooks(self, stage):
        assert stage in ["start", "tick", "end", "msg"]
        if stage == "start":
            for hook in self.hooks:
                try:
                    hook.start(self.objects)
                except:
                    pass
        if stage == "tick":
            for hook in self.hooks:
                try:
                    hook.tick(self.objects)
                except:
                    pass
        if stage == "end":
            for hook in self.hooks:
                try:
                    hook.end(self.objects)
                except:
                    pass
        if stage == "msg":
            for hook in self.hooks:
                try:
                    hook.exchangeMsg(self.objects)
                except:
                    pass

    def run (
        self,
        *,
        ticks,
        render=False,
        render_fps=30,
        render_surface=(1000, 1000),
        render_background="black",
        record=False,
        delete_recording=True,
        headless=False
    ):
        if render: assert len(self.render_fns) > 0, "You must specify a render_fn using set_render_fn(render_fn)"
        assert len(self.objects) > 0, "You must add some objects using self.add_objects(type, objects, tick_fn)"

        if record:
            assert render, "You must render to record"
            frames = []

        if render:
            if headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            import pygame
            pygame.init()
            window = pygame.display.set_mode(render_surface, pygame.HWSURFACE | pygame.DOUBLEBUF)
            clock = None
            if render_fps is not None:
                clock = pygame.time.Clock()
        
        self.run_hooks("start")
        
        self.run_global_tick_fns()

        pbar = tqdm(total=ticks)
        step = 0
        while self.running:
            while self.pause:
                self.events()
                if render:
                    self.render(
                        window=window,
                        clock=clock,
                        fps=render_fps,
                        background=render_background
                    )
            
            self.events()

            self.run_tick_fns()
            self.run_global_tick_fns()
            
            if render:
                self.render(
                    window=window,
                    clock=clock,
                    fps=render_fps,
                    background=render_background
                )
                if record:
                    frames.append(pygame.surfarray.array3d(window))
            
            self.run_hooks("tick")
            self.run_hooks("msg") 
            
            if step >= ticks: break
            step += 1
            pbar.update(1)
            GlobalLogger.tick()
            self.render_state["_tick"] += 1
        
        pbar.close()
        if record:
            import moviepy.editor as mpy
            import tempfile

            tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            frames = np.stack(frames)

            clip = mpy.ImageSequenceClip(list(frames), fps=render_fps)
            clip.write_videofile(tf.name, fps=render_fps, codec="libx264")
            
            GlobalLogger.log_media("video", tf.name)

        self.run_hooks("end")

        if record and delete_recording:
            os.unlink(tf.name)
        