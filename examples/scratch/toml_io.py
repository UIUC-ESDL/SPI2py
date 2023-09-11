import tomli

with open("input.toml", mode="rb") as fp:
    config = tomli.load(fp)

print(config)