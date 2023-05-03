import pygame
import math

from src.simulator import GlobalConfig

def render_nodes(*, window, nodes, state):
    cfg = GlobalConfig.cfg()
    for idx, node in nodes.iterrows():
        color = cfg.render.node.colors.levels[-1]
        
        if node["level"] >= 0 and node["level"] < len(list(cfg.render.node.colors.levels)):
            color = cfg.render.node.colors.levels[int(node["level"])]
        
        if node["fault"]:
            color = cfg.render.node.colors.fault

        position = node["x"], node["y"]
        
        if not node["fault"] and cfg.render.node.show_communication_radius:
           pygame.draw.circle(window, color, position, cfg.params.rc[0], 1)

        if not node["fault"] and cfg.render.node.show_perceive_radius:
            pygame.draw.circle(window, color, position, cfg.params.rs[0], 1)

        if "display_ids" not in state or not state["display_ids"]:
            if not node["fault"]:
                pygame.draw.circle(window, color, position, cfg.render.node.size)
        else:
            font = pygame.font.SysFont("Comic Sans MS", 20)
            text_surface = font.render(f"{node.id}", False, (255, 255, 255))
            pos = pygame.math.Vector2(position)
            window.blit(text_surface, pos - pygame.math.Vector2(5, 5))

def render_agents(*, window, agents, state):
    cfg = GlobalConfig().cfg()
    for idx, agent in agents.iterrows():
        if ("display_active" in state and not state["display_active"]):
            if agent["state"] == 1:
                continue

        color = cfg.render.agent.colors.levels[-1]
        if agent["level"] >= 0 and agent["level"] < len(cfg.render.agent.colors.levels):
            color = cfg.render.agent.colors.levels[int(agent["level"])]
        
        if agent.state == 0:
            color = cfg.render.agent.colors.idle
        if agent.state == 1:
            color = cfg.render.agent.colors.active

        position = agent["x"], agent["y"]


        if cfg.render.agent.show_communication_radius:
           pygame.draw.circle(window, color, position, cfg.params.rc[1], 1)

        if cfg.render.agent.show_perceive_radius:
           pygame.draw.circle(window, color, position, cfg.params.rs[1], 1)

        

        if ("display_ids" not in state or not state["display_ids"]):
            x, y = position
            angle = agent["heading"]

            points = [
                (x, y - (cfg.render.agent.height / 2)),
                (x - (cfg.render.agent.width / 2), y + (cfg.render.agent.height /2)),
                (x, y + (cfg.render.agent.height / 4)),
                (x + (cfg.render.agent.width / 2), y + (cfg.render.agent.height / 2)),
                (x, y - (cfg.render.agent.height / 2)),
                (x, y + (cfg.render.agent.height / 4)),
            ]

            position = pygame.math.Vector2((x, y))
            rotated_points = [
                (pygame.math.Vector2(p) - position) \
                .rotate_rad(angle + math.pi/2) \
                for p in points
            ]

            translated_points = [(position + p) for p in rotated_points]

            pygame.draw.polygon(
                window,
                color,
                translated_points
            )

        if "display_ids" in state and state["display_ids"]:
            font = pygame.font.SysFont("Comic Sans MS", 20)
            text_surface = font.render(f"{agent.id}", False, (255, 255, 255))
            pos = pygame.math.Vector2(position)
            window.blit(text_surface, pos - pygame.math.Vector2(5, 5))