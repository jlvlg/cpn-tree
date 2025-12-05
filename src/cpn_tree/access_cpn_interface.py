import re
from cpn_tree.cpn import CPN
import uuid
import requests


class AccessCPNInterface:
    def __init__(self) -> None:
        self.session = str(uuid.uuid4())
        self.base_url = "http://localhost:8080/api/v2/cpn"
        self.headers = {"X-SessionId": self.session}

    def load(self, cpn: CPN):
        try:  
            r = requests.post(
                f"{self.base_url}/init",
                headers=self.headers,
                json={"complex_verify": True, "need_sim_restart": True, "xml": str(cpn)},
            )
            if (r.status_code != 200):
                raise ConnectionError('Could not connect to AccessCPN Spring server')
            return r.json()
        except:
            raise ConnectionError('Could not connect to AccessCPN Spring server')


    def start_simulator(self):
        try:
            r = requests.post(
                f"{self.base_url}/sim/init",
                headers=self.headers,
                json={"options": {}},
            )
            if (r.status_code != 200):
                raise ConnectionError('Could not connect to AccessCPN Spring server')
            return r.json()
        except:
            raise ConnectionError('Could not connect to AccessCPN Spring server')


    def step_fast_forward(self, steps=10000):
        try:
            r = requests.post(
                f"{self.base_url}/sim/step_fast_forward",
                headers=self.headers,
                json={
                    "addStep": steps,
                    "addTime": 0,
                    "amount": steps,
                    "untilStep": 0,
                    "untilTime": 0,
                },
            )
            if (r.status_code != 200):
                raise ConnectionError('Could not connect to AccessCPN Spring server')
            return r.json()
        except:
            raise ConnectionError('Could not connect to AccessCPN Spring server')

    def run(self, cpn: CPN):
        self.load(cpn)
        self.start_simulator()
        while len((r := self.step_fast_forward())["enableTrans"]):
            pass
        return self.__parse_tokens(r['tokensAndMark'])

    @staticmethod
    def __parse_tokens(tokens: list[dict]) -> dict[str, list[dict]]:
        result = {}
        for item in tokens:
            _id = item['id']
            marking = item.get('marking', '').strip()
            if marking == '' or marking == 'empty':
                result[_id] = []
            else:
                parts = [p.strip() for p in marking.split('++')]
                result[_id] = [AccessCPNInterface.__parse_token(p) for p in parts]
        return result

    @staticmethod
    def __parse_token(token: str):
        marking = re.search(r"\{([^}]*)\}", token)
        quantity = re.search(r"^\d+", token)
        if not marking or not quantity:
            return {}
        body = marking.group(1)
        quantity = int(quantity.group(0))
        return dict({
            k.strip(): AccessCPNInterface.__to_value(v.strip())
            for k, v in (pair.split("=", 1)
            for pair in body.split(","))
        }, **{'quantity': quantity })

    @staticmethod
    def __to_value(v: str):
        match v:
            case "true":
                return True
            case 'false':
                return False
            case _:
                try:
                    return float(v)
                except:
                    return v.replace('"', '')
