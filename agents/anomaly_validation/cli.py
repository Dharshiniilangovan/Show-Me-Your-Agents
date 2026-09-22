import argparse,json
from pathlib import Path
from .agent import anomaly_validation_agent

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",required=True,help="JSON file containing shared state")
    p.add_argument("--dataset",default=None)
    a=p.parse_args()
    state=json.loads(Path(a.state).read_text(encoding="utf-8"))
    print(json.dumps(anomaly_validation_agent(state,a.dataset),indent=2))
if __name__=="__main__": main()
