from gymnasium.envs.registration import register

register(
    id="PassiveWalker-v0",
    entry_point="evorob.world.envs.passive_walker:PassiveWalker",
    max_episode_steps=1000,
)

register(
    id="AntHill-v0",
    entry_point="evorob.world.envs.ant_hill:AntHillEnv",
    max_episode_steps=1000,
)

# Final-project training environments — one per terrain
register(
    id="FlatEnv-v0",
    entry_point="evorob.world.envs.eval_flat:EvalFlatEnv",
    max_episode_steps=1000,
)

register(
    id="IceEnv-v0",
    entry_point="evorob.world.envs.eval_ice:EvalIceEnv",
    max_episode_steps=1000,
)

register(
    id="HillEnv-v0",
    entry_point="evorob.world.envs.eval_hill:EvalHillEnv",
    max_episode_steps=1000,
)

# Student evaluation environment (neutral reward, any terrain XML)
register(
    id="EvalEnv-v0",
    entry_point="evorob.world.envs.eval:EvalEnv",
    max_episode_steps=1000,
)

__version__ = "0.1"