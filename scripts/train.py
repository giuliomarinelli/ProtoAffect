from proto_affect.loop import run
from proto_affect.types import RunConfig

if __name__ == "__main__":
    cfg = RunConfig(T=600)
    run(cfg)
