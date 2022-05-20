from gym import envs

env_spaces = envs.registry.all()
envs = [env.id for env in env_spaces]
print(envs)