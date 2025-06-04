import configparser

def readConfig(configPath='config.ini'):
    config = configparser.ConfigParser()
    config.read(str(configPath))

    
    
    time_steps = int(config["dataset"]["timesteps"])
    group_amt = int(config["dataset"]["groups"])
    node_amt = int(config["dataset"]["nodes"])

    distance_threshold = int(config["dataset"]["distance_threshold"])
    noise_scale = float(config["dataset"]["noise_scale"])      # frequency of the noise
    noise_strength = float(config["dataset"]["noise_strength"])      # influence of the noise gradient
    tilt_strength = float(config["dataset"]["tilt_strength"])     # constant bias per group
    boundary = int(config["dataset"]["boundary"])

    hops = int(config["dataset"]["hops"])
    min_groups = int(config["dataset"]["min_groups"])

    samples = int(config["dataset"]["samples"])

    perlin_offset_amt = float(config["dataset"]["perlin_offset_amt"])