env_name = "MultiGrid-Empty-5x5-v0"

config = DQNConfig().environment(env=env_name).training().debugging(log_level="ERROR")

ppo = DQN(config)

while True:
    ppo.train()
